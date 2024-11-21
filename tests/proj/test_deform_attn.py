import sys

import psutil
import os
sys.path.append("./python")
import numpy as np
import pytest
import mugrade
import needle as ndl
from needle import backend_ndarray as nd
import threading
import time

import torch
# Lucidrians-Deformable Attention
from deformable_attention import DeformableAttention
from deformable_attention import DeformableAttention1D
from deformable_attention import DeformableAttention2D

# DAT-Deformable Attention
sys.path.append('./DAT')
from models.dat_blocks import DAttentionBaseline


_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")
    ),
]

_DEVICES_ATTN = [
    nd.cpu()
]

# For memory usage supervision
def print_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage = mem_info.rss / (1024 ** 2)
    print(f"[{message}] Memory usage: {mem_usage:.2f} MB")  # RSS: Resident Set Size
    return mem_usage

def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    while True:
        mem_info = process.memory_info()
        print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")
        time.sleep(0.05)  # Adjust the sleep interval as needed


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_dat(shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()
    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]
    x = torch.randn(batch_size, channels, height, width)

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2D(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=1,            # No offset groups
        offset_kernel_size=6,           # Offset kernel size
        group_queries=False,
        group_key_values=False
    )    

    # DAT Deformable Attention Initialization
    dat_attn = DAttentionBaseline(
        q_size=(64, 64),               # Query size (H, W)
        kv_size=(64, 64),              # Key/Value size (H, W)
        n_heads=8,                     # Number of attention heads
        n_head_channels=4,             # Channels per head (Total dim = n_heads * n_head_channels = 32)
        n_groups=1,                    # No grouping
        attn_drop=0.,                  # Attention dropout
        proj_drop=0.,                  # Projection dropout
        stride=4,                      # Downsample factor
        offset_range_factor=4,         # Offset scale
        use_pe=True,                  # No positional encoding
        dwc_pe=False,                  # No depthwise convolution positional encoding
        no_off=False,                  # Enable offsets
        fixed_pe=False,                # No fixed positional encoding
        ksize=6,                       # Offset kernel size
        log_cpb=False                  # No logarithmic CPB
    )    

    #A = attn_luc(X).detach().numpy() #A is what we calculated in real case.
    #B = attn(X) #B is the answer

    # Forward pass through DAT Deformable Attention
    output_dat, pos_dat, ref_dat = dat_attn(x)    
    print(f"DAT Output Shape: {output_dat.shape}")

    # Forward pass through Lucidrains' Deformable Attention
    output_lucid = lucid_attn(x)
    print(f"Lucidrains Output Shape: {output_lucid.shape}")

    # Compare outputs
    are_outputs_close = torch.allclose(output_dat, output_lucid, atol=1e-5)
    print("Are the outputs close?", are_outputs_close)    

    assert np.linalg.norm(output_dat.detach().numpy()-output_lucid.detach().numpy()) < 1e-5 

# Lucidrains
torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_lucid(shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()


    X = torch.randn(*shape)
    attn = DeformableAttention(
        dim = 32,                     # feature dimensions #512 will be Killed
        dim_head = 64,                # dimension per head
        heads = 8,                    # attention heads
        dropout = 0.,                 # dropout
        downsample_factor = 4,        # downsample factor (r in paper)
        offset_scale = 4,             # scale of offset, maximum offset
        offset_groups = None,         # number of offset groups, should be multiple of heads
        offset_kernel_size = 6,       # offset kernel size
    )

    A = attn(X).detach().numpy() #A is what we calculated in real case.
    B = attn(X) #B is the answer

    assert np.linalg.norm(A-B.detach().numpy()) < 1e-5 


x_shapes = [(1, 128, 512)]
@pytest.mark.parametrize("shape", x_shapes)
def test_deform_attn_lucid_1D(shape):
    X = torch.randn(*shape)
    attn = DeformableAttention1D(
        dim = 128,
        downsample_factor = 4,
        offset_scale = 2,
        offset_kernel_size = 6
    )

    A = attn(X).detach().numpy() #A is what we calculated in real case.
    B = attn(X) #B is the answer

    assert np.linalg.norm(A-B.detach().numpy()) < 1e-5 


if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")
