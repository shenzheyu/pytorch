from typing import Optional, Tuple

import torch
import tqdm

import add_dropout_fuse_cuda

class AddDropoutFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, bias1, residual1, input2, bias2, residual2, prob, training):
        # bias1 = bias1.expand_as(input1)
        # print("bias1", bias1)
        # bias2 = bias2.expand_as(input2)
        if bias1 is not None and bias2 is not None:
            output1, mask1, output2, mask2 = torch._C._nn.native_add_dropout_add_fuse(
                input1, bias1, residual1, input2, bias2, residual2, prob, training
            )
        else:
            output1, mask1, output2, mask2 = torch._C._nn.native_add_dropout_fuse(
                input1, residual1, input2, residual2, prob, training
            )
            # output1, mask1, output2, mask2 = add_dropout_fuse_cuda.forward(
            #     input1, residual1, input2, residual2, prob, training
            # )
        scale = 1.0 / (1.0 - prob)
        ctx.save_for_backward(mask1, mask2)
        ctx.scale = scale
        ctx.with_bias = bias1 is not None and bias2 is not None
        return output1, output2

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        (mask1, mask2) = ctx.saved_tensors
        scale = ctx.scale
        with_bias = ctx.with_bias
        if with_bias:
            grad_input1, grad_bias1, grad_residual1, grad_input2, grad_bias2, grad_residual2 = (
                torch._C._nn.native_add_dropout_add_fuse_2(grad_output1, mask1, grad_output2, mask2, scale)
            )
        else:
            grad_input1, grad_residual1, grad_input2, grad_residual2 = (
                torch._C._nn.native_add_dropout_fuse_2(grad_output1, mask1, grad_output2, mask2, scale)
            )
            # grad_input1, grad_residual1, grad_input2, grad_residual2 = (
            #     add_dropout_fuse_cuda.backward(grad_output1.contiguous(), mask1.contiguous(), grad_output2.contiguous(), mask2.contiguous(), scale)
            # )
            grad_bias1 = None
            grad_bias2 = None
        return grad_input1, grad_bias1, grad_residual1, grad_input2, grad_bias2, grad_residual2, None, None


class AddDropoutFuse(torch.nn.Module):
    def __init__(self):
        super(AddDropoutFuse, self).__init__()

    def forward(self, input1, bias1, residual1, input2, bias2, residual2, prob, training):
        return AddDropoutFuseFunction.apply(input1, bias1, residual1, input2, bias2, residual2, prob, training)


def bias_dropout_add1(x, bias, residual, x1, bias1, residual1, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, Tensor, Optional[Tensor], Tensor, float, bool) -> Tuple[Tensor, Tensor]
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    if bias1 is not None:
        x1 = x1 + bias1
    out1 = torch.nn.functional.dropout(x1, p=prob, training=training)
    out1 = residual1 + out1
    return out, out1

@torch.jit.script
def bias_dropout_add_fused_train1(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 x1: torch.Tensor, 
                                 bias1: Optional[torch.Tensor], 
                                 residual1: torch.Tensor,
                                 prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    return bias_dropout_add1(x, bias, residual, x1, bias1, residual1, prob, True)

class AddDropout(torch.nn.Module):
    def __init__(self):
        super(AddDropout, self).__init__()

    def forward(self, input1, bias1, residual1, input2, bias2, residual2, prob, training):
        return bias_dropout_add_fused_train1(input1, bias1, residual1, input2, bias2, residual2, prob)


import time

import torch

sequence_length = 4
batch_size = 2
hidden_size = 4

device = torch.device("cuda:7")

input1 = torch.randn(sequence_length, batch_size, hidden_size, device=device, requires_grad=True)
# bias1 = torch.randn(hidden_size, device=device, requires_grad=True)
# bias1 = torch.randn(sequence_length, batch_size, hidden_size, device=device, requires_grad=True)
bias1 = None
residual1 = torch.randn(sequence_length, batch_size, hidden_size, device=device, requires_grad=True)
input2 = torch.randn(sequence_length, batch_size, hidden_size, device=device, requires_grad=True)
# bias2 = torch.randn(hidden_size, device=device, requires_grad=True)
# bias2 = torch.randn(sequence_length, batch_size, hidden_size, device=device, requires_grad=True)
bias2 = None
residual2 = torch.randn(sequence_length, batch_size, hidden_size, device=device, requires_grad=True)
prob = 0.5
training = True
add_dropout_fuse_test = AddDropoutFuse()
add_dropout = AddDropout()

# set seed for torch
torch.manual_seed(0)

# forward = 0
# backward = 0
# for i in tqdm.tqdm(range(100)):
#     start = time.time()
#     output1, output2 = add_dropout(input1, bias1, residual1, input2, bias2, residual2, prob, training)
#     if i > 20:
#         forward += time.time() - start

#     start = time.time()
#     (output1.sum() + output2.sum()).backward()
#     if i > 20:
#         backward += time.time() - start

# print(f"Native Forward: {forward:.3f} s | Backward {backward:.3f} s")

# forward = 0
# backward = 0
# for i in tqdm.tqdm(range(100)):
#     start = time.time()
#     output1, output2 = add_dropout_fuse_test(input1, bias1, residual1, input2, bias2, residual2, prob, training)
#     if i > 20:
#         forward += time.time() - start

#     start = time.time()
#     (output1.sum() + output2.sum()).backward()
#     if i > 20:
#         backward += time.time() - start

# print(f"Fuse Forward: {forward:.3f} s | Backward {backward:.3f} s")

output1, output2 = add_dropout(input1, bias1, residual1, input2, bias2, residual2, prob, training)
output11, output22 = add_dropout_fuse_test(input1, bias1, residual1, input2, bias2, residual2, prob, training)


print("input1: ", input1)
print("bias1: ", bias1)
print("residual1: ", residual1)

print("output1: ", output11)
print("output1 gt: ", output1)

print(torch.max(output1 - output11))
print(torch.max(output2 - output22))
