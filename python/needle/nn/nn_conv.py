"""The module.
"""
from typing import List, Callable, Any
import needle as ndl
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

class ConvGp(Module):
    """
    Multi-channel 2D convolutional layer supporting group convolution
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        # Number of channels per group
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        ### BEGIN YOUR SOLUTION
        # Initialize weight tensor with Kaiming Uniform
        receptive_field = kernel_size*kernel_size
        fan_in_recept = self.in_channels_per_group*receptive_field
        #fan_out_recept = self.out_channels_per_group*receptive_field
        fan_out_recept = self.out_channels*receptive_field

        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=fan_in_recept,
                fan_out=fan_out_recept,
                shape=(out_channels, self.in_channels_per_group, kernel_size, kernel_size),
                device=device, dtype=dtype, requires_grad=True
            )
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
        """
        Forward pass for group convolution.
        Args:
            x (Tensor): Input tensor of shape (N, C_in, H, W).
        Returns:
            Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        N, C_in, H, W = x.shape
        assert C_in == self.in_channels

        # Transform input from NCHW -> NHWC
        x_nhcw = ops.transpose(x, (1, 2))
        x_nhwc = ops.transpose(x_nhcw, (2, 3))  # Shape: (N, H, W, C)

        # Calculate the appropriate padding to ensure input and output dimensions are the same
        padding = (self.kernel_size-1)//2

        # Split input into groups along the channel axis
        x_groups = ops.split(x_nhwc, axis=3)  # Each has shape (N, H, W, 1)

        # Split weights into groups along the output channel axis
        w_groups = ops.split(self.weight, axis=0)  # Each has shape (1, in_channels_per_group, K, K)

        group_outputs = []

        # Perform convolution for each group
        for group_idx in range(self.groups):
            # Stack the required slices for x_group
            x_group = ops.stack(
                [x_groups[group_idx * self.in_channels_per_group + i] for i in range(self.in_channels_per_group)],
                axis=3,
            )  # Shape: (N, H, W, in_channels_per_group)


            # Stack the required slices for w_group
            w_group = ops.stack(
                [w_groups[group_idx * self.out_channels_per_group + i] for i in range(self.out_channels_per_group)],
                axis=0,
            )  # Shape: (out_channels_per_group, in_channels_per_group, K, K)

            # Reshape weight for NHWC format
            w_group_kiok = ops.transpose(w_group, (0, 2))  # Shape: (K, in_channels_per_group, out_channels_per_group, K)
            w_group_kkoi = ops.transpose(w_group_kiok, (1, 3))  # Shape: (K, K, out_channels_per_group, in_channels_per_group)
            w_group_kkio = ops.transpose(w_group_kkoi, (2, 3))  # Shape: (K, K, out_channels_per_group, in_channels_per_group)

            # Perform convolution using ops.conv
            group_out = ops.conv(a=x_group, b=w_group_kkio, stride=self.stride, padding=padding)  # Shape: (N, H_out, W_out, out_channels_per_group)
            group_outputs.append(group_out)

        # Stack group outputs along the last dimension (channels)
        out_nhwc_list = []
        for group_out in group_outputs:
            x_groups = ops.split(group_out, axis=3)  # Each has shape (N, H, W, 1)
            for x in x_groups:
                out_nhwc_list.append(x)

        out_nhwc = ops.stack(out_nhwc_list, axis=3) #(N, H_out, W_out, C_out)

        # Add bias if present
        if self.bias:
            bias_broadcast = ops.reshape(self.bias, (1, 1, 1, self.out_channels))  # Shape: (1, 1, 1, C_out)
            out_nhwc = out_nhwc + ops.broadcast_to(bias_broadcast, out_nhwc.shape)# Shape: (N, H, W, C_out)

        # Transform output back from NHWC -> NCHW
        out_nhcw = ops.transpose(out_nhwc, (2, 3))  # Shape: (N, H, C_out, W)
        out_nchw = ops.transpose(out_nhcw, (1, 2))  # Shape: (N, C_out, H_out, W_out)

        return out_nchw