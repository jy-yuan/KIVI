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
template <int bit, int PACK_FACTOR>
__global__ void gemv_kernel_g64(
  const float4* _inputs, const uint32_t* weight, const half* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 64;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    const int num_groups_packed = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    const int weight_w = IC / PACK_FACTOR;
    // consistent with input shape
    const int sf_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2 * PACK_FACTOR;
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w, sf_w);
    int elem_per_th = 128 / bit;
    int ng_per_warp = 32 * elem_per_th / 64;
    // tile size: 4 OC x (128 * PACK_FACTOR) IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed / 2; packed_group_idx++){
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load (64,32,16) int-(2,4,8) numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      //  1 warp == 32 threads
      // g64: 1 threads -> 64,32,16 numbers -> 1,.5,0.25 group; 1 warp = 32,16,8 groups.
      // TODO: from here
      // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && packed_group_idx == 0) printf("%d %d\n", elem_per_th, ng_per_warp);
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * ng_per_warp + (threadIdx.x*ng_per_warp/32)]);
      float current_zeros = __half2float(zeros[oc_idx * sf_w + packed_group_idx * ng_per_warp + (threadIdx.x*ng_per_warp/32)]);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      const int num = 0xFF >> (8-bit);
      // multiply (64,32,16) weights with (64,32,16) inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load (16,8,4) inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          // TODO: bug is here!! for 4-bit, one float4 == 8 half number == packed_inputs[8]
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over (16,8,4) numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & num);
            float dequantized_weight = scaling_factor * current_single_weight_fp + current_zeros;
            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 1 && packed_group_idx == 0) printf("%f %f %f %f %f\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, __half2float(packed_inputs[ic_1]));
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> bit;
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
template <int bit, int PACK_FACTOR>
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
      const int num = 0xFF >> (8-bit);
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
            float current_single_weight_fp = (float)(current_packed_weight & num);
            float dequantized_weight = scaling_factor * current_single_weight_fp + current_zeros;
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> bit;
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
    if (bit == 2){
      if (group_size == 64)
      {
        gemv_kernel_g64<2, 16><<<num_blocks, num_threads>>>(
          // pointers
          in_feats, kernel, zeros, scaling_factors, out_feats,
          // constants
          num_in_channels, num_out_channels
        );
      }
      else if (group_size == 128)
      {
        gemv_kernel_g128<2, 16><<<num_blocks, num_threads>>>(
          // pointers
          in_feats, kernel, zeros, scaling_factors, out_feats,
          // constants
          num_in_channels, num_out_channels
        );}
    }else if (bit == 4){
      if (group_size == 64)
      {
        gemv_kernel_g64<4, 8><<<num_blocks, num_threads>>>(
          // pointers
          in_feats, kernel, zeros, scaling_factors, out_feats,
          // constants
          num_in_channels, num_out_channels
        );
      }
      else if (group_size == 128)
      {
        gemv_kernel_g128<4, 8><<<num_blocks, num_threads>>>(
          // pointers
          in_feats, kernel, zeros, scaling_factors, out_feats,
          // constants
          num_in_channels, num_out_channels
        );
      };} 
    else{
      if (group_size == 64)
      {
        gemv_kernel_g64<8, 4><<<num_blocks, num_threads>>>(
          // pointers
          in_feats, kernel, zeros, scaling_factors, out_feats,
          // constants
          num_in_channels, num_out_channels
        );
      }
      else if (group_size == 128)
      {
        gemv_kernel_g128<8, 4><<<num_blocks, num_threads>>>(
          // pointers
          in_feats, kernel, zeros, scaling_factors, out_feats,
          // constants
          num_in_channels, num_out_channels
        );   
      }}
    return _out_feats;
}

/*
Computes GEMV (group_size = 64).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC // PACK_FACTOR, IC;
  output: vector of shape [OC];
  zeros: matrix of shape [OC // group_size, IC];
  scaling_factors: matrix of shape [OC // group_size, IC];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g64_outer_dim(
  const float4* _inputs, const uint32_t* weight, const half* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 64;
    float psum = 0;
    const int pack_factor = 8;
    const int batch_idx = blockIdx.z;
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int group_idx = packed_oc_idx * pack_factor / group_size; 
    const float4* inputs = _inputs + batch_idx * IC;
    half* outputs = _outputs + batch_idx * OC;
    const int TILE_DIM = 32;
    extern __shared__ uint32_t packed_weight_shared[TILE_DIM];
    extern __shared__ float scale_shared[TILE_DIM];
    extern __shared__ float mn_shared[TILE_DIM];
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      if (packed_oc_idx * pack_factor < OC && k*TILE_DIM+threadIdx.x < IC)
        packed_weight_shared[threadIdx.x] = weight[packed_oc_idx * IC + k * TILE_DIM + threadIdx.x];
      else
        packed_weight_shared[threadIdx.x] = 0;
      if (group_idx * group_size < OC && k*TILE_DIM+threadIdx.x < IC){
        scale_shared[threadIdx.x] = __half2float(scaling_factors[oc_idx / group_size * IC + k * TILE_DIM + threadIdx.x]);
        mn_shared[threadIdx.x] = __half2float(zeros[oc_idx / group_size * IC + k * TILE_DIM + threadIdx.x]);
      }
      else{
        scale_shared[threadIdx.x] = 0.0;
        mn_shared[threadIdx.x] = 0.0;
      } 
      __syncthreads();
    }
}
