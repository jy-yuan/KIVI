import torch
import numpy as np
import random
from rope_embedding import fused_rope_and_quant, _fused_rope_embedding, rope_ref
from new_pack import triton_pack_along_last_dim, triton_quantize_and_pack_along_last_dim, quant_and_pack_kcache

if __name__ == '__main__':
    # position_ids.shape: [1, 744]
    # q.shape: [1, 32, 744, 128]
    # cos.shape: [744, 128]
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    BS, NH, SL, HD = 1, 32, 8192, 128
    GS = 32
    BITS = 2

    # Create embedding and Q, K, V
    position_ids = torch.arange(SL).unsqueeze(0).cuda()
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rotary_emb = LlamaRotaryEmbedding(dim=HD,max_position_embeddings=SL, base=50000).cuda()
    Q = torch.randn(BS, SL, NH, HD).half().cuda().transpose(1, 2)
    K = torch.randn(BS, SL, NH, HD).half().cuda().transpose(1, 2) # BS, NH, SL, HD
    V = torch.randn(BS, SL, NH, HD).half().cuda().transpose(1, 2)
    cos, sin = rotary_emb(Q, seq_len=SL)

    # Test 
    Q_our, K_our, v_mn, v_scale, k_mn, k_scale = fused_rope_and_quant(Q, K, V, cos, sin, position_ids,BITS, BITS, GS)
    torch.cuda.synchronize()

    # Assert
    K_input = K.transpose(1,2).view(BS, SL, NH*HD)
    K_rope_quant_ref, k_scale_ref, k_mn_ref = triton_quantize_and_pack_along_last_dim(K.permute(0,1,3,2).contiguous(), GS, BITS)
    breakpoint()
    k_mn_ref = k_mn_ref.transpose(2,3).contiguous() # BS, NH, NG, HD

    _, _, v_mn_ref = triton_quantize_and_pack_along_last_dim(V, GS, BITS)
    diff_v = ((v_mn_ref - v_mn) / v_mn_ref).abs()
    print("diff_V_mn: ", diff_v.max())

    # diff_k = ((k_mn_ref - k_mn) / k_mn_ref).abs()
    diff_k = k_mn_ref - k_mn
    print(diff_k)
    print(f"diff_K: {diff_k.max()}")
