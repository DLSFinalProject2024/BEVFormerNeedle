import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

        #self.mlp = nn.Linear(dim, heads // offset_groups, bias=False)
        #self.mlp1 = nn.ReLU()
        #self.mlp2 = nn.Linear(in_features=2, out_features=4, bias=False)

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
        #test
        pos_back = pos
        bias_back = bias
        b, i, j, c = bias.shape
        bias_to = torch.reshape(bias, (bias.shape[0] * bias.shape[1] * bias.shape[2], bias.shape[3])) #(b*i*j, c)

        for layer in self.mlp:
            bias = layer(bias)

        #test
        bias_from = bias

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias, pos_back, bias_back, bias_to, bias_from

# main class

class DeformableAttention2DLocal(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 5,
        group_queries = True,
        group_key_values = True,
        conv_qkv_bias = True,
        conv_out_bias = True
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'

        # Only to make sure padding = (offset_kernel_size - downsample_factor)/2 is integer.
        # Since I always set 'same' padding, padding = (offset_kernel_size -1)/2, I do not need this assertion.
        #assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        self.to_offsets_test = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, bias=True, padding=(offset_kernel_size-1)//2),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, bias=True, padding=(offset_kernel_size - 1) // 2),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        '''
        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2), #original setting
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )
        '''

        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = conv_qkv_bias)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = conv_qkv_bias)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = conv_qkv_bias)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = conv_out_bias)

    def forward(self, x, return_vgrid=False, return_offsets=False, return_norm_vgrid=False, return_kv_feat=False, return_pos_encoding=False, return_attn=False, return_bias_only=False, return_orig_q=False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

        # queries

        #test
        orig_x = x
        q = self.to_q(x)
        #test
        orig_q = q

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        grouped_queries = group(q)
        #test
        offsets_test = self.to_offsets_test(grouped_queries)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid =create_grid_like(offsets)
        vgrid = grid + offsets

        vgrid_scaled = normalize_grid(vgrid)

        #test
        group_x = group(x)

        kv_feats = F.grid_sample(
            group(x),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        #test
        kv_feat_orig = kv_feats

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        #test
        k_test, v_test, q_test = k, v, q

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        #test
        sim_test = sim

        # relative positional bias

        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim = 0)
        #test
        grid_x = grid
        grid_x_scaled = grid_scaled

        #test
        rel_pos_bias, pos_back, bias_back, bias_to, bias_from = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        #test
        sim_test2 = sim

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        #test
        attn_test = attn

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)


        if return_bias_only:
            return kv_feat_orig, pos_back, bias_back, bias_to, bias_from

        if return_attn:
            return kv_feat_orig, sim_test2, attn_test, out

        if return_pos_encoding:
            return kv_feat_orig, vgrid_scaled, grid_x, grid_x_scaled, rel_pos_bias, sim_test2, pos_back, bias_back, bias_to, bias_from

        if return_kv_feat:
            return kv_feat_orig, group_x, vgrid_scaled, kv_feats, k_test, v_test, q_test, sim_test

        if return_norm_vgrid:
            return offsets, vgrid_scaled

        if return_offsets:
            return grouped_queries, offsets_test

        if return_vgrid:
            return out, vgrid, orig_q, orig_x, grouped_queries

        if return_orig_q:
            return orig_q, grouped_queries

        return out
