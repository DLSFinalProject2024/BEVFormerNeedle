import needle.nn as nn
import needle.ops as ops
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
        

class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_head=12, dim_head=128, hidden_size=3072, dropout=0., device=None, dtype="float32"):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dim_head = dim_head
        self.hidden_size = hidden_size
        
        self.layer1 = nn.Sequential(
            nn.AttentionLayer(
                q_features=embed_dim,
                num_head=num_head,
                dim_head=dim_head,
                out_features=embed_dim,
                dropout=dropout,
                causal=False,
                device=device, 
                dtype=dtype
            ),
            nn.Dropout(dropout),
        )

        self.layernorm1d = nn.LayerNorm1d(embed_dim, device=device, dtype=dtype)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, device=device, dtype=dtype),
            nn.ReLU(),  # Note: Could be changed to GELU
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch_size, seq_len, x_dim = x.shape

        _x = x
        x = x.reshape((batch_size * seq_len, x_dim))
        x = self.layernorm1d(x)
        x = x.reshape((batch_size, seq_len, x_dim))
        x = self.layer1(x) + x

        _x = x
        x = x.reshape((batch_size * seq_len, x_dim))
        x = self.layernorm1d(x)
        x = x.reshape((batch_size, seq_len, x_dim))
        x = self.mlp(x)
        x += _x

        return x
    