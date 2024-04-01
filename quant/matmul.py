import torch
# import ipdb
import random
import triton
import triton.language as tl
import kivi_gemv 


@triton.jit
def qbvm_kernel(
	bits,
	a_ptr, b_ptr, c_ptr,
	scales_ptr, zeros_ptr,
	M, N, K,
	stride_abatch, stride_am, stride_ak,
	stride_bbatch, stride_bk, stride_bn,
	stride_cbatch, stride_cm, stride_cn,
	stride_scales_b, stride_scales_k, stride_scales_g,
	stride_zeros_b, stride_zeros_k, stride_zeros_g,
	groupsize,
	BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
	"""
	Compute the batch matrix multiplication C = A x B.
	A is of shape (B, 1, K) float16
	B is of shape (B, K, N//feat_per_int) int32
	C is of shape (B, 1, N) float16
	scales is of shape (B, K, G) float16
	zeros is of shape (B, K, G) float16
	groupsize is an int specifying the size of groups for scales and zeros.
	G is N // groupsize.
	Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.

	WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
	WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
	WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
	"""
	pid_batch = tl.program_id(axis=0)
	pid = tl.program_id(axis=1)
	feat_per_int = 32 // bits
	num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
	pid_n = pid % num_pid_n
	offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	a_batch_offset = (pid_batch * stride_abatch)
	b_batch_offset = (pid_batch * stride_bbatch)
	c_batch_offset = (pid_batch * stride_cbatch)
	a_ptr = a_ptr + a_batch_offset 
	b_ptr = b_ptr + b_batch_offset 
	c_ptr = c_ptr + c_batch_offset
	a_ptrs = a_ptr + (offs_k[:, None] * stride_ak)   # (BLOCK_SIZE_K, 1)
	# a_mask = (offs_am[:, None] < M)
	# b_ptrs is set up such that it repeats elements along the N axis feat_per_int times
	b_ptrs = b_ptr  + (offs_k[:, None] * stride_bk + (offs_bn[None, :]//feat_per_int) * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
	# shifter is used to extract the # bits bits of each element in the 32-bit word from B
	shifter = (offs_bn % feat_per_int) * bits
	scales_ptr = scales_ptr + pid_batch*stride_scales_b + ((offs_bn[None, :] // groupsize)) * stride_scales_g   # (BLOCK_SIZE_N,)
	zeros_ptr = zeros_ptr + pid_batch*stride_zeros_b + ((offs_bn[None, :] // groupsize)) * stride_zeros_g   # (BLOCK_SIZE_N,)

	# Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
	# M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
	# So this loop is along the infeatures dimension (K)
	# It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel	
	# accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
	num = 0xFF >> (8-bits)
	for pid_k in range(0, num_pid_k):
		offs_bk = (offs_k[:, None] + pid_k * BLOCK_SIZE_K)
		# offs_k[None, :] < K - pid_k * BLOCK_SIZE_K
		a = tl.load(a_ptrs, mask=offs_bk < K, other=0.)   # (1, BLOCK_SIZE_K)
		b = tl.load(b_ptrs, mask=offs_bk < K, other=0.)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
		ptr = scales_ptr + offs_bk * stride_scales_k 
		scales = tl.load(ptr, mask=offs_bk < K, other=0.)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
		ptr = zeros_ptr + offs_bk * stride_zeros_k  
		zeros = tl.load(ptr, mask=offs_bk < K, other=0.)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
		# Now we need to unpack b into 32-bit values
		# tl.device_print("scale ",scales.dtype)
		# tl.device_print("zeros ",zeros.dtype)
		b = (b >> shifter[None, :]) & num  # For 4-bit values, bit_op_num is 0xF
		b = b * scales + zeros # Scale and shift
		accumulator += tl.sum(a * b, 0) # tl.dot(a, b)
		# if pid_m == 0 and pid_n == 0:
		# 	tl.device_print("hello ", tl.dot(a, b).shape)
		a_ptrs += BLOCK_SIZE_K * stride_ak
		b_ptrs += BLOCK_SIZE_K * stride_bk
	c = accumulator # .to(tl.float16)
	# c = accumulator
	# Store the result
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + stride_cn * offs_cn
	c_mask = (offs_cn < N)
	tl.store(c_ptrs, c, mask=c_mask)


def understand_code():
	M, N, K = 512, 256, 256
	BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M = 64, 64, 4
	total_program_id = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
	for pid in range(0, total_program_id):
		num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
		num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
		num_pid_in_group = GROUP_SIZE_M * num_pid_n
		group_id = pid // num_pid_in_group
		first_pid_m = group_id * GROUP_SIZE_M
		group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
		pid_m = first_pid_m + (pid % group_size_m)
		pid_n = (pid % num_pid_in_group) // group_size_m
		print(f"pid={pid}, pid_m={pid_m}, pid_n={pid_n}")
	

def triton_bmm_fA_qB_outer(group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = query x key.
	Where key is quantized into 2-bit values.

	fA is of shape (B, nh, M, K) float16
	qB is of shape (B, nh, K, N // feat_per_int) int32
	scales is of shape (B, nh, K, G) float16
	zeros is of shape (B, nh, K, G) float16

	groupsize is the number of outer dimensions in each group.
	G = N // groupsize

	Returns C of shape (B, nh, M, N) float16
	"""    
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh, M, K = fA.shape 
	feat_per_int = 32 // bits
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K)
	N = qB.shape[-1] * feat_per_int
	qB = qB.reshape(-1, K, qB.shape[-1])
	# This is based on the possible BLOCK_SIZE_Ks
	# assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
	# This is based on the possible BLOCK_SIZE_Ns
	assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
	# This is based on the possible BLOCK_SIZE_Ks
	assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
	flatten_B = B * nh
	c = torch.empty((flatten_B, M, N), device='cuda', dtype=torch.float16)
	# print(f'M {M} N {N} K {K}')
	grid = lambda META: (
		flatten_B, triton.cdiv(N, META['BLOCK_SIZE_N']),
	)
	scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1])
	zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1])
	if N > K:
		BLOCK_SIZE_N = 128	
		BLOCK_SIZE_K = 32
		num_warps=4  #
	else:
		BLOCK_SIZE_N = 32
		BLOCK_SIZE_K = 128
		num_warps = 2
	num_stages= 7 if K > 64 else 3  #
	qbvm_kernel[grid](
		bits, 
		fA, qB, c,
		scales, zeros,
		M, N, K,
		fA.stride(0), fA.stride(1), fA.stride(2), 
		qB.stride(0), qB.stride(1), qB.stride(2),
		c.stride(0), c.stride(1), c.stride(2),
		scales.stride(0), scales.stride(1), scales.stride(2),
		zeros.stride(0), zeros.stride(1), scales.stride(2),
		group_size, BLOCK_SIZE_N, BLOCK_SIZE_K, 
		num_warps=num_warps, num_stages=num_stages
	)
	return c.view(B, nh, c.shape[-2], c.shape[-1])


def cuda_bmm_fA_qB_outer(group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int,
				mqa: bool=False) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = query x key.
	Where key is quantized into 2-bit values.

	fA is of shape (B, nh, M, K) float16
	qB is of shape (B, nh, K, N // feat_per_int) int32
	scales is of shape (B, nh, K, G) float16
	zeros is of shape (B, nh, K, G) float16

	groupsize is the number of outer dimensions in each group.
	G = N // groupsize

	Returns C of shape (B, nh, M, N) float16
	"""    
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh, M, K = fA.shape 
	feat_per_int = 32 // bits
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	N = qB.shape[-1] * feat_per_int
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	# This is based on the possible BLOCK_SIZE_Ks
	# assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
	# This is based on the possible BLOCK_SIZE_Ns
	# assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
	# This is based on the possible BLOCK_SIZE_Ks
	# assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
	flatten_B = B * nh
	if mqa:
		flatten_B = B
	scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	assert bits in [2, 4]
	c = kivi_gemv.gemv_forward_cuda_outer_dim(fA, qB, scales, zeros, bits, group_size, nh, mqa)
	c = c.view(B, nh, c.shape[-2], c.shape[-1])
	return c
