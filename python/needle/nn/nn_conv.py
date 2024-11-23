"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        # Initialize weights
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = init.kaiming_uniform(fan_in, fan_out, shape=weight_shape, nonlinearity="relu", device=device, dtype=dtype)
        self.weight = Parameter(self.weight, device=device, dtype=dtype) # shape: (K, K, in_channels, out_channels)
        self.bias = None # shape: (out_channels, )
        if bias:
            bias_bound = 1.0 / (fan_in ** 0.5)
            self.bias = init.rand(out_channels, low=-bias_bound, high=bias_bound, device=device, dtype=dtype)
            self.bias = Parameter(self.bias, device=device, dtype=dtype)


    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W) -> (N, H, W, C)
        nhwc_x = ops.permute(x, (0, 2, 3, 1))
        out = ops.conv(nhwc_x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias:
            out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
        # out: (N, ~H, ~W, out_channels) -> (N, out_channels, ~H, ~W)
        out = ops.permute(out, (0, 3, 1, 2))
        return out