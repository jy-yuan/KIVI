import numpy as np
import random
import triton
import triton.language as tl
import torch
from timeit_v2 import py_benchmark
from new_pack import triton_pack_along_last_dim

next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE = 65536
ROPE_GROUP_SIZE = 4


def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_ref(q, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


@triton.jit
def _rope_embedding_ref(
    Q, Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    row_position = tl.program_id(0)
    head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen) * sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen) * cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
    Q_offset = row_position*Q_row_stride + head_position*head_dim
    Q1 = tl.load(Q + Q_offset + \
                 half_head_dim*0 + col_offsets, mask = mask, other = 0).to(sin1.dtype)
    Q2 = tl.load(Q + Q_offset + \
                 half_head_dim*1 + col_offsets, mask = mask, other = 0).to(sin1.dtype)
    tl.store(Q + Q_offset + \
             half_head_dim*0 + col_offsets,
             Q1*cos1 - Q2*sin1, mask = mask)
    tl.store(Q + Q_offset + \
             half_head_dim*1 + col_offsets,
             Q2*cos1 + Q1*sin1, mask = mask)
    

@triton.jit
def _fused_rope_embedding(
    Q, K, V,
    Q_b_stride, Q_s_stride,
    K_b_stride, K_s_stride,
    V_b_stride, V_s_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    total_V_elements: tl.constexpr,
    seqlen: tl.constexpr,
    group_size: tl.constexpr,
    V_scale, V_mn,
    k_bit        : tl.constexpr,
    v_bit        : tl.constexpr,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    row_position = tl.program_id(0)
    sid = tl.program_id(1)
    seq_len_position = sid * group_size + tl.arange(0, group_size)
    head_position = tl.program_id(2)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask_col = col_offsets < half_head_dim
    mask_row = seq_len_position < seqlen
    mask = mask_row[:, None] & mask_col[None, :]
    # tl.device_print('seq_len_position', seq_len_position[:, None].shape)
    sin1 = tl.load(sin + seq_len_position[:, None] * sin_row_stride + \
                   half_head_dim*0 + col_offsets[None, :], mask = mask, other = 0)
    cos1 = tl.load(cos + seq_len_position[:, None] * cos_row_stride + \
                   half_head_dim*0 + col_offsets[None, :], mask = mask, other = 0)
    # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
    offset = row_position*Q_b_stride + seq_len_position[:, None]*Q_s_stride + head_position*head_dim
    Q1 = tl.load(Q + offset + \
                 half_head_dim*0 + col_offsets[None, :], mask = mask, other = 0).to(sin1.dtype)
    Q2 = tl.load(Q + offset + \
                 half_head_dim*1 + col_offsets[None, :], mask = mask, other = 0).to(sin1.dtype)
    tl.store(Q + offset + \
             half_head_dim*0 + col_offsets[None, :],
             Q1*cos1 - Q2*sin1, mask = mask)
    tl.store(Q + offset + \
             half_head_dim*1 + col_offsets[None, :],
             Q2*cos1 + Q1*sin1, mask = mask)
    K1 = tl.load(K + offset + \
                 half_head_dim*0 + col_offsets[None, :], mask = mask, other = 0).to(sin1.dtype)
    K2 = tl.load(K + offset + \
                 half_head_dim*1 + col_offsets[None, :], mask = mask, other = 0).to(sin1.dtype)
    tl.store(K + offset + \
             half_head_dim*0 + col_offsets[None, :],
             K1*cos1 - K2*sin1, mask = mask)
    tl.store(K + offset + \
             half_head_dim*1 + col_offsets[None, :],
             K2*cos1 + K1*sin1, mask = mask)
    
    # Calculate V scale and mn
    num_V_groups = head_dim // group_size
    ov_base = row_position*V_b_stride + seq_len_position[:, None]*V_s_stride + head_position*head_dim
    for i in range(num_V_groups):
        offsets_V =  ov_base + i * group_size + tl.arange(0, group_size)[None, :]
        cV = tl.load(V + offsets_V, mask=offsets_V<total_V_elements)
        mx_val = tl.max(cV, axis=1)
        mn_val = tl.min(cV, axis=1)
        offset_v_mn_scale = row_position * Q_b_stride // group_size + \
                            seq_len_position * Q_s_stride // group_size + \
                            head_position * head_dim // group_size
        tl.store(V_mn+offset_v_mn_scale+i, mn_val, mask=offset_v_mn_scale<total_V_elements//group_size-i)
        scale = (mx_val - mn_val) / (2 ** v_bit - 1)
        tl.store(V_scale+offset_v_mn_scale+i, scale, mask=offset_v_mn_scale<total_V_elements//group_size-i)
        
        
def get_mi_scale_ref(x, group_size, num_heads, bits):
    assert len(x.shape) == 3
    bs, seqlen, hidden_size = x.shape
    assert hidden_size % group_size == 0, print(hidden_size, group_size)
    ng = hidden_size // group_size
    x = x.view(bs, seqlen, ng, group_size)
    mi, mx = x.min(dim=-1).values, x.max(dim=-1).values
    mi = mi.view(bs, seqlen, num_heads, ng//num_heads)
    mx = mx.view(bs, seqlen, num_heads, ng//num_heads)
    scale = (mx - mi) / (2 ** bits - 1)

    return mi, scale


def fused_rope_and_quant(Q, K, V, cos, sin, position_ids, k_bit, v_bit, group_size):
    # Q = [Q1, Q2]
    # rope(Q) = [Q1 * cos - sin * Q2, Q2 * cos + sin * Q1]
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    if position_ids is not None:
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    cos, sin = cos.squeeze(), sin.squeeze()
    batch, q_seq_len, n_heads, head_dim = Q.shape
    k_seq_len, v_seq_len = K.shape[1], V.shape[1]
    Q = Q.view(batch, q_seq_len, n_heads*head_dim)
    K = K.view(batch, k_seq_len, n_heads*head_dim)
    V = V.view(batch, v_seq_len, n_heads*head_dim)
    assert(q_seq_len <= cos.shape[0])
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    assert head_dim % group_size == 0
    V_scale = torch.empty((batch, v_seq_len, n_heads, head_dim//group_size), device=V.device, dtype=V.dtype)
    V_mn = torch.empty((batch, v_seq_len, n_heads, head_dim//group_size), device=V.device, dtype=V.dtype)
    _fused_rope_embedding[(batch, triton.cdiv(q_seq_len, group_size), n_heads)](
        Q, K, V, 
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        cos, cos.stride(0),
        sin, sin.stride(0),
        V.numel(),
        q_seq_len, group_size, 
        V_scale, V_mn, k_bit, v_bit,
        head_dim, n_heads,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    # v_mi_ref, v_scale_ref = get_mi_scale_ref(V, group_size, n_heads, v_bit)
    # assert torch.allclose(v_mi_ref, V_mn)
    # assert torch.allclose(v_scale_ref, V_scale)
    return Q.view(batch, q_seq_len, n_heads, head_dim).transpose(1, 2), \
        K.view(batch, k_seq_len, n_heads, head_dim).transpose(1, 2), \
        V_mn.transpose(1, 2).contiguous(), V_scale.transpose(1, 2).contiguous()


def fast_rope_ref(Q, cos, sin, position_ids):
    Q = Q.transpose(1, 2)
    if position_ids is not None:
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    cos, sin = cos.squeeze(), sin.squeeze()
    batch, seq_len, n_heads, head_dim = Q.shape
    Q = Q.view(batch * seq_len, n_heads*head_dim)
    assert(seq_len <= cos.shape[0])
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    _rope_embedding_ref[(Q.shape[0], n_heads, )](
        Q, Q.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seq_len,
        head_dim, n_heads,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    return Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)


def fused_rope_and_quant_prefill(Q, K, V, cos, sin, position_ids, k_bit, v_bit, r, group_size):
    if V.shape[-2] <= r:
        v_quant = None
        v_full = V
        v_scale = None
        v_mn = None
        Q = fast_rope_ref(Q, cos, sin, position_ids)
        K = fast_rope_ref(K, cos, sin, position_ids)
    else:
        v_quant = V[:, :, :-r, :]
        v_full = V[:, :, -r:, :]
        Q, K, v_mn, v_scale = fused_rope_and_quant(Q, K, v_quant, cos, sin, position_ids, k_bit, v_bit, group_size)
        v_quant = triton_pack_along_last_dim(v_quant, 
                                            v_mn, v_scale,
                                            group_size, 
                                            v_bit)
    return Q, K, v_quant, v_full, v_mn, v_scale


if __name__ == "__main__":
    # position_ids.shape: [1, 744]
    # q.shape: [1, 32, 744, 128]
    # cos.shape: [744, 128]
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    BS, NH, SL, HD = 1, 32, 4096, 128
    GS = 32
    BITS = 2
    position_ids = torch.arange(SL).unsqueeze(0).cuda()
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rotary_emb = LlamaRotaryEmbedding(dim=HD,max_position_embeddings=SL, base=50000).cuda()
    Q = torch.randn(BS, SL, NH, HD).half().cuda().transpose(1, 2)
    K = torch.randn(BS, SL, NH, HD).half().cuda().transpose(1, 2)
    V = torch.randn(BS, SL, NH, HD).half().cuda().transpose(1, 2)
    cos, sin = rotary_emb(Q, seq_len=SL)
    Q_ref = rope_ref(Q, cos, sin, position_ids)
    Q_our, K_our = fused_rope_and_quant(Q, K, V, cos, sin, position_ids,BITS, BITS, GS)
    # Q_our = fused_K_rope_ref(Q, cos, sin, position_ids)
    diff = ((Q_ref - Q_our) / Q_ref).abs()
    assert not Q_ref.isnan().any()
    assert not Q_our.isnan().any()
    stmt = "rope_ref(Q, cos, sin, position_ids)"
    t_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
    stmt = "fused_rope_and_quant(Q, K, V, cos, sin, position_ids,BITS, BITS, GS)"
    # stmt = "fused_K_rope_ref(Q, cos, sin, position_ids)"
    t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
    print(diff)
    print(f"t_ref: {t_ref}, t_our: {t_our}, diff: {diff.max()}")