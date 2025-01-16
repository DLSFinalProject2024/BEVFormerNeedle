import sys
sys.path.append('./python')
sys.path.append('./apps')

import numpy as np
import pytest

import needle as ndl
import needle.nn as nn

from simple_ml import *


np.random.seed(3)


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

_DEVICES_VIT = [ndl.cpu()]


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("img_size, patch_size, in_channels, embed_dim", [
    ([224, 224], 16, 3, 768),
    ([32, 32], 4, 3, 64), # Cifar10
])
@pytest.mark.parametrize("device", _DEVICES_VIT, ids=["cpu"])
def test_patch_embedding(batch_size, img_size, patch_size, in_channels, embed_dim, device):
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
    x = np.random.rand(batch_size, in_channels, img_size[0], img_size[1])  # B, C, H, W
    x = ndl.Tensor(x, device=device)
    
    patch_embed = nn.PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    out = patch_embed(x)
    
    assert out.shape == (batch_size, num_patches, embed_dim), f"Expected shape {(batch_size, num_patches, embed_dim)}, got {out.shape}"



@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len, embed_dim, num_head, dim_head, hidden_size", [
    # (197, 768, 12, 128, 3072),  # ImageNet
    (65, 16, 1, 16, 64),  # Cifar10
])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("device", _DEVICES_VIT, ids=["cpu"])
def test_vision_transformer_block(batch_size, seq_len, embed_dim, num_head, dim_head, hidden_size, dropout, device):
    x = np.random.rand(batch_size, seq_len, embed_dim)
    x = ndl.Tensor(x, device=device)

    block = nn.VisionTransformerBlock(
        embed_dim=embed_dim,
        num_head=num_head,
        dim_head=dim_head,
        hidden_size=hidden_size,
        dropout=dropout,
        device=device
    )
    out = block(x)
    assert out.shape == (batch_size, seq_len, embed_dim), f"Expected shape {(batch_size, seq_len, embed_dim)}, got {out.shape}"



@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, dim_head, mlp_hidden_dim", [
    # ([224, 224], 16, 3, 1000, 768, 12, 128, 3072),
    ([32, 32], 4, 3, 10, 16, 1, 16, 64), # Cifar10
])
@pytest.mark.parametrize("num_blocks", [6])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("device", _DEVICES_VIT, ids=["cpu"])
def test_vision_transformer(batch_size, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, dim_head, mlp_hidden_dim, num_blocks, dropout, device):
    x = np.random.rand(batch_size, in_channels, img_size[0], img_size[1])
    x = ndl.Tensor(x, device=device)

    model = nn.VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dim_head=dim_head,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        device=device
    )
    out = model(x)
    assert out.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"

    y_onehot = np.zeros((batch_size, num_classes))
    y_onehot[np.arange(batch_size), np.random.randint(num_classes, size=batch_size)] = 1
    y_onehot = ndl.Tensor(y_onehot, device=device)

    loss = softmax_loss(out, y_onehot)
    loss.backward()

if __name__ == "__main__":
    pytest.main()