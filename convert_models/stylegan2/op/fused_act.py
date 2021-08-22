
# from torch import nn
# import torch.nn.functional as F
# # from torch.autograd import Function
# # from torch.utils.cpp_extension import load
#
#
# def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
#     return scale * F.leaky_relu(input + bias.view((1, -1)+(1,)*(len(input.shape)-2)), negative_slope=negative_slope)
#
#
# class FusedLeakyReLU(nn.Module):
#     def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
#         super().__init__()
#
#         self.bias = nn.Parameter(torch.zeros(channel))
#         self.negative_slope = negative_slope
#         self.scale = scale
#
#     def forward(self, input):
#         return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

import paddle.nn as nn
import paddle.nn.functional as F


class FusedLeakyReLU(nn.Layer):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = self.create_parameter((channel,), default_initializer=nn.initializer.Constant(0.0))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (len(input.shape) - len(bias.shape) - 1)
        return (
                F.leaky_relu(
                    input + bias.reshape((1, bias.shape[0], *rest_dim)), negative_slope=0.2
                )
                * scale
        )

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale
