import torch
import dequant_cuda
from pack import quantize_and_pack


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


def dequantize_weight(qweight, d_out, d_in, w_bit, scales, zeros, group_size):
    data = qweight.reshape(-1)
    N, num_features = d_out, d_in
    weight_fp = dequant_cuda.unpack_single_precision(data, w_bit, scales, zeros, N, 
                                                        num_features // group_size, group_size)
    return weight_fp.view(d_out, d_in)

    
class MatMul4Bit(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, qweight, bias, d_out, d_in, w_bit, scales, zeros, group_size):
        # default of pytorch behavior if inputs are empty
        # 1. Dequantize
        # 2. MatmulnN
        weight_fp = dequantize_weight(qweight, d_out, d_in, w_bit, scales, zeros, group_size)
        output = torch.nn.functional.linear(A, weight_fp.to(A.dtype), bias)
        # 3. Save state
        ctx.state = (d_out, d_in, w_bit, scales, zeros, group_size)
        ctx.tensors = qweight
        return output


    @staticmethod
    def backward(ctx, grad_output):
        req_gradA, _, req_gradBias = ctx.needs_input_grad[:3]
        qweight = ctx.tensors
        d_out, d_in, w_bit, scales, zeros, group_size = ctx.state

        grad_A, grad_bias = None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        #if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA: 
            weight_fp = dequantize_weight(qweight, d_out, d_in, w_bit, scales, zeros, group_size)
            grad_A = torch.matmul(grad_output, weight_fp.to(grad_output.dtype))
            if grad_A.isnan().any():
                import ipdb; ipdb.set_trace()
            # print(grad_A.norm())
        return grad_A, None, grad_bias, None, None, None, None, None, None
    

class WQLinearForTrain(torch.nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = (32 // self.w_bit)
        self.register_buffer('qweight', torch.zeros((out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('zeros', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size) * pack_num), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None


    def forward(self, x):
        # weight_fp = self.dequantize_weight().half()
        # out = torch.matmul(x, weight_fp.T)
        # out = out + self.bias if self.bias is not None else out

        out = MatMul4Bit.apply(x, self.qweight, self.bias, 
                               self.out_features, self.in_features, 
                               self.w_bit, self.scales, 
                               self.zeros, self.group_size)
        return out

    def dequantize_weight(self):
        data = self.qweight.reshape(-1)
        N, num_features = self.out_features, self.in_features
        weight_fp = dequant_cuda.unpack_single_precision(data, self.w_bit, self.scales, self.zeros, N, 
                                                         num_features // self.group_size, self.group_size)
        return weight_fp.view(self.out_features, self.in_features)
    

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        q_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return q_linear
        quantized, scales, mn = quantize_and_pack(linear.weight, group_size, w_bit, simulate=False)
        q_linear.qweight = quantized
        q_linear.scales = scales
        q_linear.zeros = mn
        return q_linear