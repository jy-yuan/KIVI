import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import numpy as np
import torch
import ipdb
import random
import triton
import triton.language as tl
from new_pack import pack_tensor
from timeit_v2 import py_benchmark
import kivi_gemv

B, nh, IC, OC = 8, 32, 739, 128

@triton.jit
def gemv_kernel_g64(inputs_ptr, qw_ptr, mn_ptr, 
					scale_ptr, output_ptr,
					IC: tl.constexpr, OC: tl.constexpr, bit: tl.constexpr, 
					OC_PER_PH: tl.constexpr, PACK_FACTOR: tl.constexpr, BLOCK_SIZE):
	"""
	Computes GEMV (group_size = 64).

	Args:
	inputs: vector of shape [batch_size, IC];
	qw: matrix of shape [OC, IC / 8];
	output: vector of shape [OC];
	mn: matrix of shape [OC, NG];
	scale: matrix of shape [OC, NG];

	Notes:
	One cannot infer group_size from the shape of scaling factors.
	the second dimension is rounded up to a multiple of PACK_FACTOR.
	"""
	group_size = 64
	oc_idx = tl.program_id(axis=0) * OC_PER_PH + tl.arange(0, OC_PER_PH)
	batch_idx = tl.program_id(axis=1)
	num_groups = IC // group_size
	num_groups_packed = tl.cdiv(num_groups, PACK_FACTOR)
	# tl.store(output_ptr, num_groups_packed)
	weight_w = IC // PACK_FACTOR
	num = 0xFF >> (8-bit)
	accumulator = tl.zeros((OC_PER_PH,), dtype=tl.float32)
	for group_idx in range(0, num_groups):
		# load scaling factors
		# each time we load 4 OC x 1 G
		scale = tl.load(scale_ptr + oc_idx[:, None] * num_groups + group_idx)
		mn = tl.load(mn_ptr + oc_idx[:, None] * num_groups + group_idx)
		# 1 G -> 64 numbers -> 64 // PACK_FACTOR packed numbers 
		cur_qw_ptr = qw_ptr + oc_idx[:, None] * weight_w + group_idx * (64 // PACK_FACTOR) + tl.arange(0, 64 // PACK_FACTOR)[None, :]
		qw = tl.load(cur_qw_ptr)
		for i in range(PACK_FACTOR):
			w_fp = qw & num
			# load 4 OC x 
			w_fp = w_fp * scale + mn
			qw = qw >> bit
			cur_inp_ptr = inputs_ptr + batch_idx * IC + group_idx * 64 + i + tl.arange(0, 64 // PACK_FACTOR)[None, :] * PACK_FACTOR
			cur_input = tl.load(cur_inp_ptr)
			accumulator += tl.sum(cur_input * w_fp, 1)
	ptr = output_ptr + oc_idx + batch_idx * OC
	tl.store(ptr, accumulator)


def dequant_weight(w, scale, mn, gs):
	w_fp = w.half().view(w.shape[0], w.shape[1]//gs, gs)
	w_fp = w_fp * scale.unsqueeze(-1) + mn.unsqueeze(-1)
	return w_fp.view(w.shape)


def dequant_weight_outer(w, scale, mn, gs):
	# ipdb.set_trace()
	w_fp = w.half().view(w.shape[0], w.shape[1], w.shape[2]//gs, gs)
	w_fp = w_fp * scale.unsqueeze(-1) + mn.unsqueeze(-1)
	return w_fp.view(w.shape)


def gemv_fwd(bit, group_size, inp, qweight, mn, scale):
	B, IC = inp.shape
	OC = qweight.shape[0]
	BLOCK_SIZE = 32
	OC_PER_PH = 32
	PACK_FACTOR = 32 // bit
	assert group_size == 64
	output = torch.empty((B, OC), device=inp.device, dtype=torch.float16)
	grid = lambda META: (
		triton.cdiv(OC, META['OC_PER_PH']), B
	)
	gemv_kernel_g64[grid](inp, qweight, mn, scale, output, 
					   IC, OC, bit, OC_PER_PH, PACK_FACTOR, BLOCK_SIZE)
	return output


def test_bgemv_outer_correct_mha():
	flatten_B = B * nh
	inp = torch.randn((flatten_B, 1, IC), device='cuda', dtype=torch.float16)
	ori_weight = torch.randn((flatten_B, IC, OC), device='cuda', dtype=torch.float16)
	GS = 32
	for BIT in [2, 4]:
		weight = ori_weight
		PACK_FACTOR = 32 // BIT
		assert OC % GS == 0 and OC % PACK_FACTOR == 0
		NG = OC // GS
		weight = weight.view(flatten_B, IC, NG, GS)
		mx = torch.max(weight, dim=-1, keepdim=False)[0]
		mn = torch.min(weight, dim=-1, keepdim=False)[0]
		maxq = 2 ** BIT - 1
		scale = (mx - mn) / maxq
		weight = weight - mn.unsqueeze(-1)
		weight.div_(scale.unsqueeze(-1))
		weight = weight.clamp_(0, maxq).round_().to(torch.int32)
		weight = weight.view(flatten_B, IC, OC)
		qweight = pack_tensor(weight, BIT, 2)
		weight = weight.transpose(1, 2).contiguous()
		qweight = qweight.transpose(1, 2).contiguous()
		scale = scale.transpose(1, 2).contiguous()
		mn = mn.transpose(1, 2).contiguous()
		output = kivi_gemv.gemv_forward_cuda_outer_dim(inp, qweight, scale, mn, BIT, GS, nh, False)
		deq_w = dequant_weight_outer(weight.transpose(1, 2), 
							   scale.transpose(1, 2), 
							   mn.transpose(1, 2), GS)
		# rel_error = torch.abs((deq_w - ori_weight).float() / (ori_weight + 1e-5).float()).mean()
		# print(f'bit {BIT} avg rel weight quant error: {rel_error}')
		output_ref = inp @ deq_w
		error = output_ref - output
		rel_out_error = torch.abs(error.float() / (torch.abs(output_ref).float()+1e-5)).mean()
		print(f'mha bit {BIT} avg rel out quant error: {rel_out_error}')


def test_bgemv_outer_correct_mqa():
	flatten_B = B * nh
	inp = torch.randn((flatten_B, 1, IC), device='cuda', dtype=torch.float16)
	ori_weight = torch.randn((B, IC, OC), device='cuda', dtype=torch.float16)
	GS = 32
	for BIT in [2, 4]:
		weight = ori_weight
		PACK_FACTOR = 32 // BIT
		assert OC % GS == 0 and OC % PACK_FACTOR == 0
		NG = OC // GS
		weight = weight.view(B, IC, NG, GS)
		mx = torch.max(weight, dim=-1, keepdim=False)[0]
		mn = torch.min(weight, dim=-1, keepdim=False)[0]
		maxq = 2 ** BIT - 1
		scale = (mx - mn) / maxq
		weight = weight - mn.unsqueeze(-1)
		weight.div_(scale.unsqueeze(-1))
		weight = weight.clamp_(0, maxq).round_().to(torch.int32)
		weight = weight.view(B, IC, OC)
		qweight = pack_tensor(weight, BIT, 2)
		inp = inp.contiguous()
		weight = weight.transpose(1, 2).contiguous()
		qweight = qweight.transpose(1, 2).contiguous()
		scale = scale.transpose(1, 2).contiguous()
		mn = mn.transpose(1, 2).contiguous()
		output = kivi_gemv.gemv_forward_cuda_outer_dim(inp, qweight, scale, mn, BIT, GS, nh, True)
		deq_w = dequant_weight_outer(weight.transpose(1, 2), 
							   scale.transpose(1, 2), 
							   mn.transpose(1, 2), GS)
		# rel_error = torch.abs((deq_w - ori_weight).float() / (ori_weight + 1e-5).float()).mean()
		# print(f'bit {BIT} avg rel weight quant error: {rel_error}')
		output_ref = inp.view(B, nh, 1, IC) @ deq_w.view(B, 1, IC, OC)
		output_ref = output_ref.view(flatten_B, 1, OC)
		error = output_ref - output
		# ipdb.set_trace()
		rel_out_error = torch.abs(error.float() / (torch.abs(output_ref).float()+1e-5)).mean()
		print(f'mqa bit {BIT} avg rel out quant error: {rel_out_error}')


def test_gemv_correct():
	inp = torch.randn((B, IC), device='cuda', dtype=torch.float16) 
	ori_weight = torch.randn((OC, IC), device='cuda', dtype=torch.float16) 
	GS = 64
	for BIT in [4]:
		weight = ori_weight
		PACK_FACTOR = 32 // BIT
		assert IC % GS == 0 and IC % PACK_FACTOR == 0
		NG = IC // GS
		weight = weight.view(OC, NG, GS)
		mx = torch.max(weight, dim=2, keepdim=False)[0]
		mn = torch.min(weight, dim=2, keepdim=False)[0]
		maxq = 2 ** BIT - 1
		scale = (mx - mn) / maxq
		weight = weight - mn.unsqueeze(-1)
		weight.div_(scale.unsqueeze(-1))
		weight = weight.clamp_(0, maxq).round_().to(torch.int32)
		weight = weight.view(OC, IC)
		qweight = pack_tensor(weight, BIT, 1)
		# output = gemv_fwd(BIT, GS, inp, qweight, mn, scale)
		output = kivi_gemv.gemv_forward_cuda(inp, qweight, scale, mn, BIT, GS)
		deq_w = dequant_weight(weight, scale, mn, GS)
		rel_error = torch.abs((deq_w - ori_weight).float() / (ori_weight + 1e-5).float()).mean()
		# print(f'bit {BIT} avg rel weight quant error: {rel_error}')
		output_ref = inp @ deq_w.T
		error = output_ref - output
		rel_out_error = torch.abs(error.float() / (output_ref + 1e-5).float()).mean()
		print(f'bit {BIT} avg rel out quant error: {rel_out_error}')


def test_gemv_speed():
	inp = torch.randn((B, IC), device='cuda', dtype=torch.float16)
	ori_weight = torch.randn((OC, IC), device='cuda', dtype=torch.float16)
	weight = ori_weight
	BIT = 4
	GS = 64
	PACK_FACTOR = 32 // BIT
	assert IC % GS == 0 and IC % PACK_FACTOR == 0
	NG = IC // GS
	weight = weight.view(OC, NG, GS)
	mx = torch.max(weight, dim=2, keepdim=False)[0]
	mn = torch.min(weight, dim=2, keepdim=False)[0]
	maxq = 2 ** BIT - 1
	scale = (mx - mn) / maxq
	weight = weight - mn.unsqueeze(-1)
	weight.div_(scale.unsqueeze(-1))
	weight = weight.clamp_(0, maxq).round_().to(torch.int32)
	weight = weight.view(OC, IC)
	qweight = pack_tensor(weight, BIT, 1)
	output = gemv_fwd(BIT, GS, inp, qweight, mn, scale)
	deq_w = dequant_weight(weight, scale, mn, GS)
	stmt = "inp @ deq_w.T"
	t_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
	# stmt = "gemv_fwd(BIT, GS, inp, qweight, mn, scale)"
	# t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
    #                                  setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
	stmt = "kivi_gemv.gemv_forward_cuda(inp, qweight, scale, mn, BIT, GS)"
	t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
	print(f'vanilla pytorch gemv: {t_ref * 1000} ms')
	print(f'awq fused IC {IC} OC {OC} {BIT}-bit gemv: {t_our * 1000} ms')


def test_bgemv_outer_speed():
	inp = torch.randn((B, 1, IC), device='cuda', dtype=torch.float16)
	ori_weight = torch.randn((B, IC, OC), device='cuda', dtype=torch.float16) 
	GS = 64
	for BIT in [2]:
		weight = ori_weight
		PACK_FACTOR = 32 // BIT
		assert OC % GS == 0 and OC % PACK_FACTOR == 0
		NG = OC // GS
		weight = weight.view(B, IC, NG, GS)
		mx = torch.max(weight, dim=-1, keepdim=False)[0]
		mn = torch.min(weight, dim=-1, keepdim=False)[0]
		maxq = 2 ** BIT - 1
		scale = (mx - mn) / maxq
		weight = weight - mn.unsqueeze(-1)
		weight.div_(scale.unsqueeze(-1))
		weight = weight.clamp_(0, maxq).round_().to(torch.int32)
		weight = weight.view(B, IC, OC)
		qweight = pack_tensor(weight, BIT, 2)
		weight = weight.transpose(1, 2).contiguous()
		qweight = qweight.transpose(1, 2).contiguous()
		scale = scale.transpose(1, 2).contiguous()
		mn = mn.transpose(1, 2).contiguous()
		deq_w = dequant_weight_outer(weight.transpose(1, 2), 
							scale.transpose(1, 2), 
							mn.transpose(1, 2), GS)
		stmt = "inp @ deq_w"
		t_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
										setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
		# stmt = "gemv_fwd(BIT, GS, inp, qweight, mn, scale)"
		# t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
		#                                  setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
		stmt = "kivi_gemv.gemv_forward_cuda_outer_dim(inp, qweight, scale, mn, BIT, GS)"
		t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
										setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
		print(f'BS {B} IC {IC} OC {OC} pytorch batched gemv: {t_ref * 1000} ms')
		print(f'our fused BS {B} IC {IC} OC {OC} {BIT}-bit outer-dim batched gemv: {t_our * 1000} ms')

if __name__ == "__main__":
	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)
	# test_gemv_correct()
	test_bgemv_outer_correct_mha()
	test_bgemv_outer_correct_mqa()
	# test_gemv_speed()
	# test_bgemv_outer_speed()
