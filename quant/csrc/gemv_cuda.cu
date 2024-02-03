// Inspired by https://github.com/ankan-ban/llama_cu_awq
/*

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

*/

#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "gemv_cuda.h"
#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define WARP_SIZE 32


// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
  return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}


/*
Computes GEMV (group_size = 64).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g64(
  const float4* _inputs, const uint32_t* weight, const half* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 64;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    // This is essentially zeros_w.
    const int num_groups_packed = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    // consistent with input shape
    const int sf_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2 * PACK_FACTOR;
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w, sf_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed / 2; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g64: two threads -> 64 numbers -> 1 group; 1 warp = 16 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);
      float current_zeros =  __half2float(zeros[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * current_single_weight_fp + current_zeros;
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (group_size = 128).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g128(
  const float4* _inputs, const uint32_t* weight, const half* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
    // consistent with input shape
    const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;
    //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
      float current_zeros = __half2float(zeros[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * current_single_weight_fp + current_zeros;
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC, IC // 8];
  _zeros: int tensor of shape [OC, IC // G // 8];
  _scaling_factors: tensor of shape [OC, IC // G];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;

Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<float4*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [OC, IC]
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(0)}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    int blockDim_z = num_out_feats;
    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
    dim3 num_threads(32, 4);
    if (group_size == 64)
    {
      gemv_kernel_g64<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels
      );
    }
    else if (group_size == 128)
    {
      gemv_kernel_g128<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels
      );
    }
    return _out_feats;
;}




/*
Computes Batched 4-bit GEMV (group_size = 64).

Args:
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  zeros: matrix of shape [BS, OC // group_size, IC];
  scaling_factors: matrix of shape [BS, OC // group_size, IC];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv4_kernel_outer_dim(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int nh, const bool mqa){
    const int bit = 4;
    const int pack_factor = 8;
    const int batch_idx = blockIdx.x;
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; 
    const half* inputs = _inputs + batch_idx * IC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx;
    if (mqa){
      _batch_idx = batch_idx / nh;
    }else{
      _batch_idx = batch_idx;
    }
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    // 1float4 == 8 half number
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // each thread load 32 int4 number
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      // each thread load 32 int4 number
      // int weight_offset = packed_oc_idx * IC + k * TILE_DIM + threadIdx.x*4;
      // if (weight_offset < OC * IC / pack_factor)
      //   *((float4*)(qw)) = *((float4*)(weight + packed_oc_idx * IC + k * TILE_DIM + threadIdx.x*4));
      // int scale_mn_offset = group_idx * IC + k * TILE_DIM + threadIdx.x*4;
      // if (scale_mn_offset < OC * IC / group_size){
      //   *((float2*)(cscale)) = *((float2*)(scaling_factors + scale_mn_offset));
      //   *((float2*)(czero)) = *((float2*)(zeros + scale_mn_offset));
      // }
      // int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      // if (inputs_ptr_delta < IC){
      //   const half* inputs_ptr = inputs + inputs_ptr_delta;
      //   *((float2*)(inp)) = *((float2*)(inputs_ptr));
      // }
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            cur_packed_weight = cur_packed_weight >> bit;
            psum[ic_1] += dequantized_weight * cur_inp;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        psum[i] = warp_reduce_sum(psum[i]);
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}


__global__ void bgemv2_kernel_outer_dim(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int nh, const bool mqa){
    // const int group_size = 64;
    const int bit = 2;
    const int pack_factor = 16;
    const int batch_idx = blockIdx.x;
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; 
    const int ICR = IC;
    const half* inputs = _inputs + batch_idx * ICR;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx;
    if (mqa){
      _batch_idx = batch_idx / nh;
    }else{
      _batch_idx = batch_idx;
    }
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    // 1float4 == 8 half number
    float psum[pack_factor]{};
    for (int k=0; k < (ICR + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // each thread load 32 int4 number
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      // if (weight_offset < OC * ICR / pack_factor)
      //   *((float4*)(qw)) = *((float4*)(weight + packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4));
      // int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      // if (scale_mn_offset < OC * ICR / group_size){
      //   *((float2*)(cscale)) = *((float2*)(scaling_factors + scale_mn_offset));
      //   *((float2*)(czero)) = *((float2*)(zeros + scale_mn_offset));
      // }
      // int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      // if (inputs_ptr_delta < ICR){
      //   const half* inputs_ptr = inputs + inputs_ptr_delta;
      //   *((float2*)(inp)) = *((float2*)(inputs_ptr));
      // }
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            cur_packed_weight = cur_packed_weight >> bit;
            psum[ic_1] += dequantized_weight * cur_inp;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        psum[i] = warp_reduce_sum(psum[i]);
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

// __global__ void bgemv2_kernel_g64_outer_dim(
//   const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
//   const int IC, const int OC){
//     const int group_size = 64;
//     const int bit = 2;
//     const int pack_factor = 16;
//     const int batch_idx = blockIdx.x;
//     const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
//     const int oc_start_idx = packed_oc_idx * pack_factor;
//     const int group_idx = oc_start_idx / group_size; 
//     const int ICR = IC;
//     const half* inputs = _inputs + batch_idx * ICR;
//     half* outputs = _outputs + batch_idx * OC;
//     const uint32_t*  weight = _weight + batch_idx * OC * IC / pack_factor;
//     const half* scaling_factors = _scale + batch_idx * OC * IC / group_size;
//     const half* zeros = _zeros + batch_idx * OC * IC / group_size;
//     const int TILE_DIM = 128;
//     const int num = 0xFF >> (8-bit);
//     // 1float4 == 8 half number
//     float psum[pack_factor]{};
//     for (int k=0; k < (ICR + TILE_DIM - 1) / TILE_DIM; k++){
//       uint32_t qw[4]{};
//       half cscale[4]{};
//       half czero[4]{};
//       half inp[4]{};
//       // each thread load 32 int4 number
//       int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
//       if (weight_offset < OC * ICR / pack_factor)
//         *((float4*)(qw)) = *((float4*)(weight + packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4));
//       int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
//       if (scale_mn_offset < OC * ICR / group_size){
//         *((float2*)(cscale)) = *((float2*)(scaling_factors + scale_mn_offset));
//         *((float2*)(czero)) = *((float2*)(zeros + scale_mn_offset));
//       }
//       int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
//       if (inputs_ptr_delta < ICR){
//         const half* inputs_ptr = inputs + inputs_ptr_delta;
//         *((float2*)(inp)) = *((float2*)(inputs_ptr));
//       }
//       // multiply 32 weights with 32 inputs
//       #pragma unroll
//       for (int ic_0 = 0; ic_0 < 4; ic_0++){
//         uint32_t cur_packed_weight =  qw[ic_0];
//         float cur_inp = __half2float(inp[ic_0]);
//         float cur_scale = __half2float(cscale[ic_0]);
//         float cur_zero = __half2float(czero[ic_0]);
//         for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
//           int oc_idx = oc_start_idx + ic_1;
//           if (oc_idx < OC){
//             float cur_single_weight_fp = (float)(cur_packed_weight & num);
//             float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
//             // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
//             cur_packed_weight = cur_packed_weight >> bit;
//             psum[ic_1] += dequantized_weight * cur_inp;
//           }
//         }
//       }
//     }
//     for (int i=0; i < pack_factor; i++){
//       int oc_idx = oc_start_idx + i;
//       if (oc_idx < OC){
//         psum[i] = warp_reduce_sum(psum[i]);
//         if (threadIdx.x == 0) 
//           outputs[oc_idx] = __float2half(psum[i]); 
//       }
//     }
// }


/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC // PACK_Factor, IC];
  _zeros: int tensor of shape [OC // G, IC];
  _scaling_factors: tensor of shape [OC // G, IC];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;
Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_forward_cuda_outer_dim(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size,
    const int nh,
    const bool mqa)
{
    int BS = _in_feats.size(0);
    int num_in_feats = _in_feats.size(1);
    int num_in_channels = _in_feats.size(2);
    int num_out_channels = _zeros.size(1) * group_size;
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [OC, IC]
    at::Tensor _out_feats = torch::empty({BS, num_in_feats, num_out_channels}, options);
    int num_out_feats = _out_feats.size(-2);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    int pack_factor = 32 / bit;
    dim3 num_blocks(BS, (num_out_channels / pack_factor + 3) / 4, num_out_feats);
    dim3 num_threads(32, 4);
    if (bit == 4){
      bgemv4_kernel_outer_dim<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels, group_size, nh, mqa
      );}
    else{
      // note: in this case, pack factor == 16
      bgemv2_kernel_outer_dim<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels, group_size, nh, mqa
      );     
      }
    return _out_feats;
;}
