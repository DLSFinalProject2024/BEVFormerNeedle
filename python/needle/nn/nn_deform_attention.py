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
    Tanh,
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

def create_grid_like(t, dim = 0):
    b, c, h, w = t.shape
    device = t.device  # Get the device of the input tensor

    # Create the grid using NumPy
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h), indexing='xy'), axis=dim)
    #grid = np.reshape(grid, (1, 2, h, w))
    #grid = np.broadcast_to(grid, (b, 2, h, w))

    # Convert to PyTorch tensor and move to the same device and dtype as `t`
    grid_tensor = Tensor(grid, device=device, dtype=t.dtype, requires_grad=False)
    return grid_tensor

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    device, dtype = grid.device, grid.dtype

    grid_h, grid_w = ops.split(grid, axis = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return ops.stack([grid_h, grid_w], axis=out_dim)    

class Scale(Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# Only support tanh approximation
class GELU(Module):
    def forward(self, X: Tensor) -> Tensor:
        x_cubed = ops.power_scalar(X, 3)  # x^3 = x * x * x
        scaled_x_cubed = ops.mul_scalar(x_cubed, 0.044715)
        z = ops.add(X, scaled_x_cubed)  # z = x + 0.044715 * x^3
        z_scaled = ops.mul_scalar(z, np.sqrt(2/np.pi))
        tanh_z = ops.tanh(z_scaled)  # Tanh(sqrt(2/pi)*(x+0.044715*x^3))
        w = ops.add_scalar(tanh_z, 1)
        gelu = ops.mul_scalar(ops.multiply(X, w), 0.5)  # GELU(x) = 0.5 * x * (1 + tanh(...))
        return gelu        

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
        offset_kernel_size = 5,
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

        # For testing
        self.kv_feats_from_luc = None

        self.device = device
        self.dtype = dtype

        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        # Only to make sure padding = (offset_kernel_size - downsample_factor)/2 is integer.
        # Since I always set 'same' padding, padding = (offset_kernel_size -1)/2, I do not need this assertion.
        #assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        self.dim = dim
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.offset_groups = offset_groups

        self.offset_dims = self.inner_dim // offset_groups

        self.downsample_factor = downsample_factor
        
        self.to_q_bias = to_q_bias
        self.to_k_bias = to_k_bias
        self.to_v_bias = to_v_bias
        self.to_out_bias = to_out_bias

        self.to_offsets = Sequential(
            ConvGp(self.offset_dims, self.offset_dims, offset_kernel_size, groups = self.offset_dims, stride = downsample_factor, bias=True, device=self.device, dtype=self.dtype),
            GELU(),
            ConvGp(self.offset_dims, 2, 1, bias = False),
            Tanh(),
            Scale(offset_scale)
        )

        #self.rel_pos_bias = CPB(self.dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)
        self.dropout = Dropout(dropout)
        self.to_q = ConvGp(self.dim, self.inner_dim, 1, groups = offset_groups if group_queries else 1, bias = self.to_q_bias, device=self.device, dtype=self.dtype)
        self.to_k = ConvGp(self.dim, self.inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = self.to_k_bias, device=self.device, dtype=self.dtype)
        self.to_v = ConvGp(self.dim, self.inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = self.to_v_bias, device=self.device, dtype=self.dtype)
        self.to_out = ConvGp(self.inner_dim, self.dim, 1, bias=self.to_out_bias, device=self.device, dtype=self.dtype)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

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
        return_vgrid=False,
        return_norm_vgrid=False,
        return_kv_feat=False,
        return_pos_encoding=False
    ):
        """
        The forward function of the Deformable Attention function.
        Input: x with shape (batch_size, in_channels, height, width), NCHW
        Output: z with shape (batch_size, in_channels, height, width), NCHW
        """

        Bin, Cin, Hin, Win, downsample_factor, device = *x.shape, self.downsample_factor, x.device

        # queries
        q = self.to_q(x) #(Bin, Cin, Hin, Win)
        _, _, h_q, w_q = q.shape

        # reshape queries into groups
        grouped_queries = q.reshape((Bin*self.offset_groups, self.offset_dims, h_q, w_q)) #(Bin*self.offset_groups, self.offset_dims, Hin, Win)

        # pass groups of queries into offset network
        offsets = self.to_offsets(grouped_queries) #(Bin*self.offset_groups, 2, Hin/downsample_factor, Win/downsample_factor)

        grid = create_grid_like(offsets) #(2, Hin/downsample_factor, Win/downsample_factor)
        grid = ops.reshape(grid, (1, grid.shape[0], grid.shape[1], grid.shape[2]))#(1, 2, Hin/downsample_factor, Win/downsample_factor)
        grid = ops.broadcast_to(grid, (offsets.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]))#(Bin*self.offset_groups, 2, Hin/downsample_factor, Win/downsample_factor)
        vgrid = grid+offsets #(Bin*self.offset_groups, 2, Hin/downsample_factor, Win/downsample_factor)
        vgrid_scaled = normalize_grid(vgrid, dim=1, out_dim=3) #(Bin*self.offset_groups, Hin/downsample_factor, Win/downsample_factor, 2)

        # sampling features
        if self.kv_feats_from_luc is not None:
            kv_feat = self.kv_feats_from_luc.reshape((Bin, self.offset_groups*self.offset_dims, offsets.shape[-2], offsets.shape[-1])) #(Bin, Cin, Hin/downsample_factor, Win/downsample_factor)
            k, v = self.to_k(kv_feat), self.to_v(kv_feat) #(Bin, Cin, Hin/downsample_factor, Win/downsample_factor)
            q = q*self.scale #(Bin, Cin, Hin/downsample_factor, Win/downsample_factor)

            #Q
            q_bin, q_cin, q_hin, q_win = q.shape
            q = q.reshape((Bin, self.heads, self.dim_head, q_hin*q_win))
            q = q.transpose((2, 3)) #(Bin, self.heads, Win/down_sample_factor*Hin/downsample_factor, self.dim_head)

            #K
            k_bin, k_cin, k_hin, k_win = k.shape
            k = k.reshape((Bin, self.heads, self.dim_head, k_hin*k_win))
            k = k.transpose((2, 3)) #(Bin, self.heads, Win/down_sample_factor*Hin/downsample_factor, self.dim_head)

            #V
            v_bin, v_cin, v_hin, v_win = v.shape
            v = v.reshape((Bin, self.heads, self.dim_head, v_hin*v_win))
            v = v.transpose((2, 3)) #(Bin, self.heads, Win/down_sample_factor*Hin/downsample_factor, self.dim_head)
        
            # similarity
            sim = self.matmul(q, k)

            # calculate relative positional encoding
            grid_x = create_grid_like(x) #(2, Hin, Win)
            grid_x_scaled = normalize_grid(grid_x, dim=0, out_dim=2) #(Hin, Win, 2)


        if return_pos_encoding:
            return vgrid_scaled, grid_x, grid_x_scaled

        if return_kv_feat:
            return kv_feat, k, v, q, sim

        if return_norm_vgrid:
            return q, grouped_queries, offsets, vgrid_scaled

        return q, grouped_queries, offsets



