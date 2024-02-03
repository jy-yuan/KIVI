import os
import random
import numpy as np
import torch
from matmul import triton_bmm


def quant_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
	assert len(k.shape) == 4
	k = k.flatten(0, 1)
	shape = k.shape
	B, T, D = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	# if T % group_size != 0:
	# 	# Padding
	# 	new_T = (T // group_size + 1) * group_size
	# 	delta = new_T - T
	# 	k = torch.cat([k, torch.zeros([B, delta, D], dtype=k.dtype, device=k.device)], 1)
	# B, nG, group_size, D
	k_groups = k.view(B, -1, group_size, D)
	mn, mx = torch.min(k_groups, 2)[0], torch.max(k_groups, 2)[0]
	# B, nG, D
	scale = (mx - mn) / (2 ** bits - 1)
	# ====================================================
	intk = []
	ng = k_groups.shape[1]
	for idx in range(ng * group_size):
		g_idx = idx // group_size
		# B, nG, D
		q = torch.round((k[:, idx, :] - mn[:, g_idx, :]) / scale[:, g_idx, :]).to(torch.int32)
		intk.append(q[:, None, :])
	intk = torch.cat(intk, 1)
	feat_per_int = 32 // bits
	code = torch.zeros([B, k.shape[1]//feat_per_int, D], dtype=torch.int32, device=k.device)
	i = 0
	row = 0
	while row < code.shape[1]:
		if bits in [2,4,8]:
			for j in range(i, i + (32 // bits)):
				code[:, row, :] |= intk[:, j, :] << (bits * (j - i))
			i += 32 // bits
			row += 1
		else:
			raise NotImplementedError("Only 2,4,8 bits are supported.")
	return code, scale, mn


def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
	shape = v.shape
	assert len(shape) == 4
	assert v.shape[-1] % group_size == 0
	num_groups = shape[-1] // group_size
	new_shape = (shape[:-1] + (num_groups, group_size))
	# Quantize
	max_int = 2 ** bits - 1
	data = v.view(new_shape)
	mn = torch.min(data, dim=-1, keepdim=True)[0]
	mx = torch.max(data, dim=-1, keepdim=True)[0]
	scale = (max_int) / (mx - mn)
	data = data - mn
	data.mul_(scale)
	data = data.clamp_(0, max_int).round_().to(torch.int32)
	data = data.view(shape)
	# Pack
	code = pack_tensor(data, bits, pack_dim=3)
	return code, scale, mn


def unpack_and_dequant_vcache(v_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	assert bits in [2, 4, 8]
	assert len(v_code.shape) == 4
	data = unpack_tensor(v_code, bits, pack_dim=3)
	shape = data.shape
	num_groups = shape[-1] // group_size
	data = data.view(shape[:-1] + (num_groups, group_size,))
	data = data.to(torch.float16)
	data = data / scale + mn 
	return data.view(shape)


def pack_tensor(data, bits, pack_dim):
	# Pack
	shape = data.shape
	feat_per_int = 32 // bits
	assert bits in [2,4,8], "Only 2, 4, 8 bits are supported"
	assert shape[pack_dim] % (32 // bits) == 0, "Dimension length must be divisible by number of features per int"
	# BS, nh, T, nd // 16 # 16 is for 2bit
	code = torch.zeros(shape[:pack_dim] + (shape[-1] // feat_per_int,)+shape[pack_dim+1:], 
					dtype=torch.int32, 
					device=data.device)
	i = 0
	row = 0
	unpacked_indices = [slice(None)] * len(data.shape)
	packed_indices = [slice(None)] * len(data.shape)
	while row < code.shape[pack_dim]:
		packed_indices[pack_dim] = row
		for j in range(i, i + (32 // bits)):
			unpacked_indices[pack_dim] = j
			code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
		i += 32 // bits
		row += 1
	return code


def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	return unpacked_v_code


def test_vcache():
	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)
	B, nh, T, hd = 555, 32, 433, 128
	v = torch.randn((B, nh, T, hd), device='cuda', dtype=torch.float16)
	group_size = 64
	for bits in [2, 4, 8]:
		code, scale, mn = quant_and_pack_vcache(v, group_size, bits)
		# print(f'bit {bits}, scale.shape: {scale.shape}')
		# print(f'bit {bits}, code.shape: {code.shape}')
		dequant_v = unpack_and_dequant_vcache(code, scale, mn, group_size, bits)
		assert not dequant_v.isnan().any()
		gap = (dequant_v - v) / v
		gap = torch.nan_to_num(gap)
		print(f'bit {bits}, mean rel arr: {torch.mean(torch.abs(gap))}')
	

def test_pack_unpack():
	torch.manual_seed(3407)
	np.random.seed(3407)
	random.seed(3407)
	# B, nh, T, hd = 555, 32, 433, 128
	B, nh, T, hd = 64, 64, 64, 64
	for pack_dim in range(4):
		for bits in (2, 4, 8):
			intensor = torch.randint(2 ** bits, (B, nh, T, hd), dtype=torch.int32, device='cuda')
			packed_code = pack_tensor(intensor, bits, pack_dim)
			unpacked_tensor = unpack_tensor(packed_code, bits, pack_dim)
			torch.allclose(intensor, unpacked_tensor.int(), atol=1e-2, rtol=0)

def test_kcache():
	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)
	query_len = 1
	BS, nh, T, D = 1, 32, 64, 128
	k = torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
	group_size = 64
	for bits in [8, 4, 2]:
		code, scale, mn = quant_and_pack_kcache(k, group_size, bits)
		import ipdb; ipdb.set_trace()

# def test_kcache():
# 	torch.manual_seed(0)
# 	np.random.seed(0)
# 	random.seed(0)
# 	query_len = 1
# 	BS, nh, T, D = 1, 32, 64, 128
# 	k = torch.randn((BS, nh, T, D), device='cuda', dtype=torch.float16)
# 	group_size = 64
# 	for bits in [8, 4, 2]:
# 		code, scale, mn = quant_and_pack_kcache(k, group_size, bits)
# 		B, _, D = code.shape
# 		# B, nh, seq_len // feat_per_int, hd
# 		code = code.view(BS, nh, -1, D)
# 		scale = scale.view(BS, nh, -1, D)
# 		mn = mn.view(BS, nh, -1, D)
# 		query_state = torch.randn((BS, nh, query_len, D), device='cuda', dtype=torch.float16)
# 		our_out = triton_bmm(group_size, query_state, code, scale, mn, bits)
# 		ref_out = torch.matmul(query_state, k.transpose(2, 3))
# 		gap = (our_out - ref_out) / ref_out
# 		err = torch.mean(torch.abs(gap)).item()
# 		print(f'bits {bits}, err: {err}')


if __name__ == '__main__':
	# test_pack_unpack()
	test_vcache()
	test_kcache()