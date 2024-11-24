import needle.nn as nn
import needle.ops as ops
from needle.nn import MultiHeadAttention
from needle.autograd import Tensor
# import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=16, in_channels=3, embed_dim=768, device=None, dtype="float32"):
        super().__init__()
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, "Image dimensions must be divisible by the patch size"
        # Note: embed_dim = 16 * 16 * 3 = 768
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # 14 * 14 = 196
        self.proj = nn.Conv(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)

    def forward(self, x):
        # (batch, in_channels, img_size, img_size) -> (batch, embed_dim, sqrt(num_patches), sqrt(num_patches))
        x = self.proj(x)

        # (batch, embed_dim, sqrt(num_patches), sqrt(num_patches)) -> (batch, embed_dim, num_patches)
        x = x.reshape((x.shape[0], self.embed_dim, self.num_patches))

        # (batch, embed_dim, num_patches) -> (batch, num_patches, embed_dim)
        x = x.permute((0, 2, 1))
        return x
        
