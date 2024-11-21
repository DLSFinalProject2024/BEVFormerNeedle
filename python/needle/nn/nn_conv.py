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

        ### BEGIN YOUR SOLUTION
        # Initialize weight tensor with Kaiming Uniform
        receptive_field = kernel_size*kernel_size
        fan_in_recept = in_channels*receptive_field
        fan_out_recept = out_channels*receptive_field

        self.weight = Parameter(
            init.kaiming_uniform(fan_in=fan_in_recept,
                                 fan_out=fan_out_recept, 
                                 shape=(kernel_size, kernel_size, in_channels, out_channels), device=device, dtype=dtype, requires_grad=True)
        )

        # Initialize bias tensor, if applicable, with Uniform distribution
        if bias:
            bound = 1.0 / np.sqrt(fan_in_recept)
            self.bias = Parameter(
                init.rand(*(out_channels,), low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True)
            )
        else:
            self.bias = None        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Ensure nn.Conv works for `(N, C, H, W)` tensors even though we implemented the conv op for `(N, H, W, C)` tensors
        N, C, H, W = x.shape
        x_nhcw = ops.transpose(x, (1, 2))
        x_nhwc = ops.transpose(x_nhcw, (2, 3))

        # Calculate the appropriate padding to ensure input and output dimensions are the same
        padding = (self.kernel_size-1)//2

        # Calculate the convolution, then add the properly-broadcasted bias term if present
        x_out_nhwc = ops.conv(a=x_nhwc, b=self.weight, stride=self.stride, padding=padding)
        if self.bias:
            bias_reshape = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            bias_broadcast = ops.broadcast_to(bias_reshape, x_out_nhwc.shape)
            x_out_nhwc = x_out_nhwc + bias_broadcast
        x_out_nhcw = ops.transpose(x_out_nhwc, (2, 3))
        x_out_nchw = ops.transpose(x_out_nhcw, (1, 2))

        return x_out_nchw
        ### END YOUR SOLUTION