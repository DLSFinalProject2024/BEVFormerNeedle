from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential,
    Residual
)

from .nn_conv import (
    ConvGp
)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

class DeformableAttention(Module):
    """
    The deformable attention module.
    """
    def __init__(
        self,
        *,
        dim=32,
        dim_head = 4,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 6,
        group_queries = True,
        group_key_values = True,
        to_q_bias = False,
        to_k_bias = False,
        to_v_bias = False,
        to_out_bias = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        self.dim = dim
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = self.inner_dim // offset_groups

        self.downsample_factor = downsample_factor
        
        self.to_q_bias = to_q_bias
        self.to_k_bias = to_k_bias
        self.to_v_bias = to_v_bias
        self.to_out_bias = to_out_bias

        '''
        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )
        '''

        #self.rel_pos_bias = CPB(self.dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)
        self.dropout = Dropout(dropout)
        self.to_q = ConvGp(self.dim, self.inner_dim, 1, groups = offset_groups if group_queries else 1, bias = self.to_q_bias, device=self.device, dtype=self.dtype)
        self.to_k = ConvGp(self.dim, self.inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = self.to_k_bias, device=self.device, dtype=self.dtype)
        self.to_v = ConvGp(self.dim, self.inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = self.to_v_bias, device=self.device, dtype=self.dtype)
        self.to_out = ConvGp(self.inner_dim, self.dim, 1, bias=self.to_out_bias, device=self.device, dtype=self.dtype)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        x,
        return_vgrid=False
    ):
        """
        The forward function of the Deformable Attention function.
        Input: x with shape (batch_size, in_channels, height, width), NCHW
        Output: z with shape (batch_size, in_channels, height, width), NCHW
        """

        heads, Bin, Cin, Hin, Win, downsample_factor, device = self.heads, *x.shape, self.downsample_factor, x.device

        # queries
        q = self.to_q(x)
        return q



