import needle.nn as nn
import needle.ops as ops
from needle.autograd import Tensor
import needle.init as init
import math
# import numpy as np


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer.
    Input: (batch, in_channels, img_size, img_size)
    Output: (batch, num_patches, embed_dim)
    """
    def __init__(self, img_size=[224, 224], patch_size=16, in_channels=3, embed_dim=768, device=None, dtype="float32"):
        super().__init__()
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, "Image dimensions must be divisible by the patch size"
        # Note: embed_dim = 16 * 16 * 3 = 768
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # 14 * 14 = 196
        # self.proj = nn.Conv(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0, device=device, dtype=dtype)
        self.linear = nn.Linear(in_channels * patch_size * patch_size, embed_dim, device=device, dtype=dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        cut_col = []
        x_list = list(ops.split(x, axis=2))
        for i in range(H // self.patch_size):
            cur = x_list[i*self.patch_size: (i+1)*self.patch_size]
            cut_col.append(ops.stack(cur, axis=2))
            
        patches = []  # (batch, in_channels, patch_size, patch_size) * num_patches
        for xx in cut_col:
            xx_list = list(ops.split(xx, axis=3))
            for i in range(W // self.patch_size):
                cur = xx_list[i*self.patch_size: (i+1)*self.patch_size]
                patches.append(ops.stack(cur, axis=3))
        
        x = ops.stack(patches, axis=1)  # (batch, num_patches, in_channels, patch_size, patch_size)
        x = x.reshape((B * self.num_patches, C * self.patch_size * self.patch_size))
        # (batch, num_patches, in_channels * patch_size * patch_size) -> (batch, num_patches, embed_dim)
        x = self.linear(x)
        x = x.reshape((B, self.num_patches, self.embed_dim))


        # Old failed implementation: Conv has issue in out settings
        # # (batch, in_channels, img_size, img_size) -> (batch, embed_dim, sqrt(num_patches), sqrt(num_patches))
        # x = self.proj(x)

        # # (batch, embed_dim, sqrt(num_patches), sqrt(num_patches)) -> (batch, embed_dim, num_patches)
        # x = x.reshape((x.shape[0], self.embed_dim, self.num_patches))

        # # (batch, embed_dim, num_patches) -> (batch, num_patches, embed_dim)
        # x = x.permute((0, 2, 1))
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
    

class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification tasks.
    """
    def __init__(
        self, img_size=[224, 224], patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, num_blocks=6,
        num_heads=12, dim_head=128, mlp_hidden_dim=3072, dropout=0.1, device=None, dtype="float32",
        deform_attn_activate=False, dattn_dim_head=1, dattn_heads=1, dattn_offset_groups=1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, device, dtype)

        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # 14 * 14 = 196
        self.cls_token = nn.Parameter(
            init.zeros(1, embed_dim, device=device, dtype=dtype, requires_grad=True)  # TODO: need grad?
        )

        # (1, 197, 768)
        self.positional_embedding = nn.Parameter(init.zeros(1, self.num_patches + 1, embed_dim, device=device, dtype=dtype))
        self.positional_dropout = nn.Dropout(dropout)
        self.deform_attn_activate = deform_attn_activate

        self.transformer_blocks = nn.Sequential(
            *[VisionTransformerBlock(
                embed_dim=embed_dim,
                num_head=num_heads,
                dim_head=dim_head,
                hidden_size=mlp_hidden_dim,
                dropout=dropout,
                device=device,
                dtype=dtype
            ) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(embed_dim, num_classes, device=device, dtype=dtype)

        
        # Ours Deformable Attention Initialization
        if self.deform_attn_activate:
            self.preproc = nn.Conv(3, in_channels, 3, stride=1, bias=True, device=device, dtype=dtype)
            self.dattn = nn.Residual(
                nn.Sequential(
                    #nn.Conv(16, 16, 3, stride=1, bias=False, device=device, dtype=dtype),
                    nn.DeformableAttention(
                        dim=in_channels,                   # Feature dimensions
                        dim_head=dattn_dim_head,                 # Dimension per head
                        heads=dattn_heads,                       # Attention heads
                        dropout=0.,                        # Dropout
                        downsample_factor=4,           # Downsample factor
                        offset_scale=4,                # Offset scale
                        offset_groups=dattn_offset_groups,   # Offset groups
                        offset_kernel_size=5,          # Offset kernel size
                        group_queries=True,
                        group_key_values=True,
                        to_q_bias = True,
                        to_k_bias = True,
                        to_v_bias = True,
                        to_out_bias = True,
                        device=device,
                        dtype='float32',
                        return_out_only=True
                    )
                )
            )


    def forward(self, x):
        """
        Input: (batch, in_channels, img_size, img_size)
        Output: (batch, num_classes)
        """
        if self.deform_attn_activate:
            x = self.preproc(x) #(batch, c=dim, image_size, image_size)
            x = self.dattn(x) #(batch, c=dim, image_size, image_size)

        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        batch_size, num_patches, embed_dim = x.shape

        # Add [CLS] token
        cls_tokens = self.cls_token.broadcast_to((batch_size, embed_dim))
        xs = tuple(ops.split(x, axis=1))
        x = (cls_tokens,) + xs
        x = ops.stack(x, axis=1)  # (batch, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.positional_embedding.broadcast_to(x.shape)

        # Transformer layers
        x = self.transformer_blocks(x)  # (batch, num_patches + 1, embed_dim)

        # Extract [CLS] token representation
        x_cls = ops.split(x, axis=1)[0]  # (batch, embed_dim)

        # Classification head
        x = self.head(x_cls)

        return x
