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


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("img_size, patch_size, in_channels, embed_dim", [
    ([224, 224], 16, 3, 768),
    ([32, 32], 4, 3, 64), # Cifar10
])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_patch_embedding(batch_size, img_size, patch_size, in_channels, embed_dim, device):
    num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
    x = np.random.rand(batch_size, in_channels, img_size[0], img_size[1])  # B, C, H, W
    x = ndl.Tensor(x, device=device)
    
    patch_embed = nn.PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    out = patch_embed(x)
    
    assert out.shape == (batch_size, num_patches, embed_dim), f"Expected shape {(batch_size, num_patches, embed_dim)}, got {out.shape}"


if __name__ == "__main__":
    pytest.main()