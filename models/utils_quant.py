# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = (
                    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        output = torch.round(input * s).div(s + 1e-6)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (
                    (
                        input.max(dim=-1, keepdim=True)[0]
                        - input.min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class AsymGroupedQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, group_size, prec_map_indices=None):
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).

        bs, seqlen, d = input.shape
        num_groups = d // group_size
        if num_groups * group_size != input.shape[-1]:
            raise ValueError("group_size should be a factor of the last dimension size")


        input_in_groups = input.view(bs, seqlen, num_groups, group_size)

        #####
        # input_in_groups_cpy = input_in_groups.clone().detach()
        #####

        mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
        mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

        scale = (mx - mn) / (2 ** num_bits - 1)
        input_in_groups = (input_in_groups - mn) / scale
        input_in_groups = F.relu(input_in_groups)
        rounded_input_in_groups = input_in_groups.round_()
        dequantized_input_in_groups = rounded_input_in_groups * scale + mn

        #####
        # if prec_map_indices is not None:
        #     _, num_heads, _ = prec_map_indices.shape
        #     for i in range(bs):
        #         for j in range(num_heads):
        #             for k in prec_map_indices[i, j]:
        #                 dequantized_input_in_groups[i, k, j, :] = input_in_groups_cpy[i, k, j, :]
        #####

        dequantized_input = dequantized_input_in_groups.view(bs, seqlen, -1)
        return dequantized_input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output

        # clip version
        # grad_input[input.ge(clip_val[1])] = 0
        # grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None

### group by channel
class AsymGroupedQuantizerByChannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, group_size, prec_map_indices=None):
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        bs, seqlen, d = input.shape
        mx, mn = input.max(dim=-2)[0], input.min(dim=-2)[0]
        mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)
        scale = (mx - mn) / (2 ** num_bits - 1)
        input = (input - mn) / scale
        input = F.relu(input)
        rounded_input = input.round_()
        dequantized_input = rounded_input * scale + mn

        assert dequantized_input.shape == input.shape

        return dequantized_input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output

        # clip version
        # grad_input[input.ge(clip_val[1])] = 0
        # grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None

class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=32,
        a_bits=32,
        act_layerwise=False,
        weight_layerwise=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise
        # params for weight quant
        # if self.w_bits < 32:
        #     self.weight_clip_val = Parameter(torch.tensor([-2.0, 2.0]), requires_grad=False)
        if self.a_bits < 32 and self.a_bits > 2:
            if symmetric:
                self.act_quantizer = SymQuantizer
            else:
                self.act_quantizer = AsymQuantizer

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 32:
            weight = self.weight
        elif self.w_bits >= 3:
            weight_clip_val = torch.tensor([-2.0, 2.0])
            weight = SymQuantizer.apply(
                real_weights, weight_clip_val, self.w_bits, self.weight_layerwise
            )
        else:
            if self.w_bits == 1:
                if self.weight_layerwise:
                    scaling_factor = torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = torch.mean(
                        abs(real_weights), dim=1, keepdim=True
                    ).detach()
                quan_weights_no_grad = scaling_factor * (
                    torch.sign(real_weights / scaling_factor)
                )
            # elif self.w_bits == 2:
            #     scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            #     quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
            else:
                num_bits = 2 ** (self.w_bits - 1)
                clip_val = 1 - 1e-2
                if self.weight_layerwise:
                    scaling_factor = 2 * torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = (
                        2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
                    )
                quan_weights_no_grad = (
                    scaling_factor
                    * (
                        torch.round(
                            torch.clamp(
                                real_weights / scaling_factor, -clip_val, clip_val
                            )
                            * num_bits
                            - 0.5
                        )
                        + 0.5
                    )
                    / num_bits
                )

            weight = (
                quan_weights_no_grad.detach() - real_weights.detach() + real_weights
            )
        # Quantize inputs
        if self.a_bits < 32 and self.a_bits > 2:
            act_clip_val = torch.tensor([-2.0, 2.0])
            input_ = self.act_quantizer.apply(
                input_, act_clip_val, self.a_bits, self.act_layerwise
            )

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


def test_group_quantize():
    input = torch.randn((4, 16, 1024), dtype=torch.float16, device='cuda')
    clip_val = torch.tensor([-2.0, 2.0])
    for num_bits, group_size in [ (2, 64), (4, 64), (8, 64), \
                                 (2, 128), (4, 128), (8, 128), \
                                 (2, 256), (4, 256), (8, 256)]:
        output = AsymGroupedQuantizer.apply(input, clip_val, num_bits, group_size)
        err = torch.mean(torch.abs(input - output)).item()
        print(num_bits, group_size, err)
    # print(input[0,0,100:150])
    # print(output[0,0,100:150])


def process_input(input, group_size):
    N = input.shape[0]
    input_flatten = input.reshape(N, -1)
    num_features = input_flatten.shape[1]

    # Compute min, max by groups
    if num_features % group_size != 0:
        # Padding
        new_num_features = (num_features // group_size + 1) * group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

    input_groups = input_flatten.reshape(-1, group_size)
    mn, mx = torch.min(input_groups, 1)[0], torch.max(input_groups, 1)[0]
    return input_groups.view(N, -1, group_size), mn.view(N, -1), mx.view(N, -1)


def quantize_and_pack(data, group_size, bits, simulate=False):
    data, mn, mx = process_input(data, group_size)
    data = data.transpose(0, 1)
    mn = mn.t()
    mx = mx.t()
    if simulate:
        mn, mx = mn.unsqueeze(-1), mx.unsqueeze(-1)
        N = data.shape[0]
        output = data   # N, groups, group_dim
        if isinstance(bits, int): 
            bits = torch.ones(N, dtype=torch.int32, device='cuda') * bits

        B = (2 ** bits - 1).view(N, 1, 1)
        mn = mn - 1e-6
        mx = mx + 1e-6
        scale = B / (mx - mn)     # N, groups, 1
        output = (output - mn) * scale
        output = F.relu(output)
        output = torch.min(output, B.float()).round_().int()
    else:
        # data.shape == B, ng, gz
        # mn.shape == B, ng
        # mx.shape == B, ng
        # import ipdb; ipdb.set_trace()
        output, scale = dequant_cuda.pack_single_precision(data, mn, mx, bits, False)
    scale = scale.squeeze(-1)
    return output, scale, mn


def dequantize_and_unpack(data, group_size, shape, bits, scale, mn, simulate=False):
    if simulate:
        scale, mn = scale.unsqueeze(-1), mn.unsqueeze(-1)
        data = data / scale + mn
    else:
        # Pad to group_size
        N = shape[0]
        num_features = int(np.prod(shape[1:]))
        num_features = (num_features + (group_size - num_features % group_size) % group_size)

        # Unpack bitstream
        data = dequant_cuda.unpack_single_precision(data, bits, scale, mn, N, num_features // group_size, group_size)
    data  = data.view(shape)
    return data


def process_input_by_channel(input, group_size):
    num_features = input.shape[-1]
    # input_flatten: [num_feats, bs * seqlen]
    input_flatten = input.view(-1, num_features).transpose(0, 1)
    num_instances = input_flatten.shape[-1]
    # Compute min, max by groups
    if num_instances % group_size != 0:
        # Padding
        new_num_instances = (num_instances // group_size + 1) * group_size
        delta = new_num_instances - num_instances
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([num_features, delta], dtype=input.dtype, device=input.device)], 1)
    input_groups = input_flatten.reshape(-1, group_size)
    mn, mx = torch.min(input_groups, 1)[0], torch.max(input_groups, 1)[0]
    return input_groups.view(num_features, -1, group_size), mn.view(num_features, -1), mx.view(num_features, -1)


def quantize_by_channel_and_pack(input, group_size, num_bits, simulate=False):
    assert len(input.shape) == 3
    shape = input.shape
    ori_num_instances = shape[0] * shape[1]
    input_groups, mn, mx = process_input_by_channel(input, group_size)
    if simulate:
        mn, mx = mn.unsqueeze(-1), mx.unsqueeze(-1)
        scale = (mx - mn) / (2 ** num_bits - 1)
        input_groups = (input_groups - mn) / scale
        input_groups = F.relu(input_groups)
        rounded_input = input_groups.round_()
        return rounded_input, scale, mn
        # dequantized_input = rounded_input * scale + mn
        # dequantized_input = dequantized_input.view(input.shape[-1], -1)
        # if ori_num_instances != dequantized_input.shape[1]:
        #     dequantized_input = dequantized_input[:, 0:ori_num_instances]
        # dequantized_input = dequantized_input.transpose(0, 1).view(shape)
        # assert dequantized_input.shape == shape
        # return dequantized_input, scale, mn
    else:
        output, scale = dequant_cuda.pack_single_precision(input_groups, mn, mx, num_bits, False)
    assert len(scale.shape) >= 2 and len(mn.shape) >= 2
    if len(scale.shape) == 3:
        scale = scale.squeeze(-1)
    if len(mn.shape) == 3:
        mn = mn.squeeze(-1)
    return output, scale, mn


def dequantize_by_channel_and_unpack(data, group_size, shape, bits, scale, mn, simulate=False):
    num_feats = shape[-1]
    ori_num_instances = shape[0] * shape[1]
    if simulate:
        # import ipdb; ipdb.set_trace()
        data = data * scale + mn
    else:
        # Pad to group_size
        tot_num_instances = (ori_num_instances + (group_size - ori_num_instances % group_size) % group_size)

        # Unpack bitstream
        data = dequant_cuda.unpack_single_precision(data, bits, scale, mn, num_feats, tot_num_instances // group_size, group_size)
    dequantized_input = data.view(shape[-1], -1)
    if ori_num_instances != dequantized_input.shape[1]:
        dequantized_input = dequantized_input[:, 0:ori_num_instances]
    data = dequantized_input.transpose(0, 1).view(shape)
    return data


def cal_tensor_size(x):
    if isinstance(x, list):
        return np.sum([cal_tensor_size(x_) for x_ in x])
    elif isinstance(x, torch.Tensor):
        num_params = np.prod(x.shape)
        if x.dtype == torch.int32:
            return num_params * 4
        elif x.dtype in [torch.bfloat16, torch.float16]:
            return num_params * 2
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    

def quantize_by_channel_and_pack_cache(input, group_size, num_bits, simulate=False):
    ## convert kv_cache shape (bsz, head, seq_len, dim_head) to zirui shape (bsz, seq_len, dim)
    assert len(input.shape) == 4
    bsz, _, seq_len, _ = input.shape
    input = input.transpose(1, 2).reshape(bsz, seq_len, -1)
    ##

    shape = input.shape
    ori_num_instances = shape[0] * shape[1]
    input_groups, mn, mx = process_input_by_channel(input, group_size)
    if simulate:
        mn, mx = mn.unsqueeze(-1), mx.unsqueeze(-1)
        scale = (mx - mn) / (2 ** num_bits - 1)
        input_groups = (input_groups - mn) / scale
        input_groups = F.relu(input_groups)
        rounded_input = input_groups.round_()
        return rounded_input, scale, mn
        # dequantized_input = rounded_input * scale + mn
        # dequantized_input = dequantized_input.view(input.shape[-1], -1)
        # if ori_num_instances != dequantized_input.shape[1]:
        #     dequantized_input = dequantized_input[:, 0:ori_num_instances]
        # dequantized_input = dequantized_input.transpose(0, 1).view(shape)
        # assert dequantized_input.shape == shape
        # return dequantized_input, scale, mn
    else:
        # import ipdb; ipdb.set_trace()
        output, scale = dequant_cuda.pack_single_precision(input_groups, mn, mx, num_bits, False)
    assert len(scale.shape) >= 2 and len(mn.shape) >= 2
    if len(scale.shape) == 3:
        scale = scale.squeeze(-1)
    if len(mn.shape) == 3:
        mn = mn.squeeze(-1)
    return output, scale, mn


def dequantize_by_channel_and_unpack_cache(data, group_size, shape, bits, scale, mn, simulate=False):
    ## the input shape is not zirui shape (bsz, seq_len, dim), but kv_cache shape (bsz, head, seq_len, dim_head)
    # original variables
    # num_feats = shape[-1]
    # ori_num_instances = shape[0] * shape[1]
    assert len(shape) == 4
    num_feats = shape[1] * shape[3]
    ori_num_instances = shape[0] * shape[2]
    ##

    if simulate:
        # import ipdb; ipdb.set_trace()
        data = data * scale + mn
    else:
        # Pad to group_size
        tot_num_instances = (ori_num_instances + (group_size - ori_num_instances % group_size) % group_size)

        # Unpack bitstream
        data = dequant_cuda.unpack_single_precision(data, bits, scale, mn, num_feats, tot_num_instances // group_size, group_size)
    dequantized_input = data.view(num_feats, -1)
    if ori_num_instances != dequantized_input.shape[1]:
        dequantized_input = dequantized_input[:, 0:ori_num_instances]
    
    ## convert zirui shape (bsz, seq_len, dim) to kv_cache shape (bsz, head, seq_len, dim_head)
    # this is last step (this shape is zirui shape) data = dequantized_input.transpose(0, 1).view(shape)
    data = dequantized_input.transpose(0, 1).view(shape[0], -1, num_feats)
    data = data.view(shape[0], shape[2], shape[1], -1).transpose(1, 2)
    ##
    assert data.shape == shape

    return data

def test_channel_quantize():
    input = torch.randn((112, 334, 4096), dtype=torch.float16, device='cuda')
    shape = input.shape
    # for num_bits, group_size in [ (2, 64), (4, 64), (8, 64), \
    #                              (2, 128), (4, 128), (8, 128), \
    #                              (2, 256), (4, 256), (8, 256)]:
    for num_bits, group_size in [(2, 128), (4, 128)]:
        # fake_code, scale, mn = quantize_by_channel_and_pack(input, group_size, num_bits, True)
        # output_fake = dequantize_by_channel_and_unpack(fake_code, group_size, shape, num_bits, scale, mn, True)
        # err = torch.mean(torch.abs(input - output_fake)).item()
        # print(num_bits, group_size, err)
        real_code, scale, mn = quantize_by_channel_and_pack(input, group_size, num_bits, False)
        output_real = dequantize_by_channel_and_unpack(real_code, group_size, shape, num_bits, scale, mn, False)
        err = torch.mean(torch.abs(input - output_real)).item()
        print(num_bits, group_size, err)

def test_quantize():
    input = torch.randn((1, 32, 340, 128), dtype=torch.float16, device='cuda')
    shape = input.shape
    quantized_v, scale, mn = quantize_and_pack(input, 128, 4, False)
    dequantized_v = dequantize_and_unpack(quantized_v, 128, shape, 4, scale, mn, False)

    quantized_v, scale, mn = quantize_by_channel_and_pack_cache(input, 128, 4, False)
    dequantized_v = dequantize_by_channel_and_unpack_cache(quantized_v, 128, shape, 4, scale, mn, False)


if __name__ == '__main__':
    # test_group_quantize()
    # test_channel_quantize()
    test_quantize()