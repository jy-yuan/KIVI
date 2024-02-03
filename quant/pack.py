import torch
import numpy as np
import torch.nn.functional as F
import dequant_cuda


def process_input(input, group_size):
    N = input.shape[0]
    input_flatten = input.view(N, -1)
    num_features = input_flatten.shape[1]

    # Compute min, max by groups
    if num_features % group_size != 0:
        # Padding
        new_num_features = (num_features // group_size + 1) * group_size
        delta = new_num_features - num_features
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

    input_groups = input_flatten.view(-1, group_size)
    mn, mx = torch.min(input_groups, 1)[0], torch.max(input_groups, 1)[0]
    return input_groups.view(N, -1, group_size), mn.view(N, -1), mx.view(N, -1)


def quantize_and_pack(data, group_size, bits, simulate=False):
    data, mn, mx = process_input(data, group_size)
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
        output, scale = dequant_cuda.pack_single_precision(data, mn, mx, bits, False)
    scale, mn = scale.squeeze(), mn.squeeze()
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
