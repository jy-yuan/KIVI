import torch
import numpy as np
import random
from rope_embedding import fused_rope_and_quant, _fused_rope_embedding, rope_ref, fast_rope_ref
from new_pack import triton_pack_along_last_dim, triton_quantize_and_pack_along_last_dim, quant_and_pack_kcache
from torch.profiler import profile, ProfilerActivity 

if __name__ == '__main__':
    # position_ids.shape: [1, 744]
    # q.shape: [1, 32, 744, 128]
    # cos.shape: [744, 128]
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    BS, NH, SL, HD = 1, 8, 8192, 64
    # BS, NH, SL, HD = 1, 32, 8192, 128
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

    # Reference: (MAKE SURE NO REFERENCE MODIFIES Q, K, V IN PLACE)
    K_rope_ref = fast_rope_ref(K.clone(), cos, sin, position_ids)
    # NOTE: reference quantization is applied AFTER ROPE has been applied
    K_quant_ref, k_scale_ref, k_mn_ref = triton_quantize_and_pack_along_last_dim(K_rope_ref.permute(0,1,3,2).contiguous(), GS, BITS)
    k_mn_ref = k_mn_ref.transpose(2,3).contiguous() # BS, NH, NG, HD
    k_scale_ref = k_scale_ref.transpose(2,3).contiguous() # BS, NH, NG, HD
    
    # Act 
    Q_rope, K_rope, v_mn, v_scale, k_mn, k_scale = fused_rope_and_quant(Q, K.clone(), V, cos, sin, position_ids,BITS, BITS, GS)
    torch.cuda.synchronize()
    torch.set_printoptions(edgeitems=4, linewidth=180, sci_mode=False, precision=6)

    # Assert
    _, _, v_mn_ref = triton_quantize_and_pack_along_last_dim(V, GS, BITS)
    diff_v = (v_mn_ref - v_mn)
    print("diff_V_mn: ", diff_v.max())

    # Assert K rope
    diff_K_rope = K_rope_ref - K_rope
    print("diff_K_rope: ", diff_K_rope.max())

    # Assert K mn
    diff_k_mn = k_mn_ref - k_mn
    print(f"diff_K_mn: {diff_k_mn.max()}")

    # Assert K scale
    diff_k_scale = (k_scale_ref - k_scale)
    print(f"diff_K_scale: {diff_k_scale}")
    print(f"diff_K_scale: {diff_k_scale.max()}")
    breakpoint()
    # Assert K quant
    k_rope_quant = triton_pack_along_last_dim(K_rope.transpose(2,3).contiguous(),
                                              k_mn.transpose(2,3).contiguous(),
                                              k_scale.transpose(2,3).contiguous(),
                                              GS, 
                                              BITS)
    diff_k_quant = (K_quant_ref - k_rope_quant)
    print(f"diff_K_quant: {diff_k_quant.max()}")
    breakpoint()    