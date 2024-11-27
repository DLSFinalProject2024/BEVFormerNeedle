import sys

import psutil
import os
sys.path.append("./python")
import numpy as np
import pytest
import needle as ndl
from needle import backend_ndarray as nd
import threading
import time

import torch
import torch.nn
# Lucidrians-Deformable Attention
sys.path.append('./LUC')
from deformable_attention import DeformableAttention
from deformable_attention import DeformableAttention1D
from deformable_attention import DeformableAttention2D
from deformable_attention_2d_local import DeformableAttention2DLocal

# DAT-Deformable Attention
sys.path.append('./DAT')
from models.dat_blocks import DAttentionBaseline

print(f"BACKEND = {ndl.BACKEND}")


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
x_shapes = [(1, 32, 64, 64), 
            (2, 32, 64, 64),
            (4, 16, 32, 32),
            (10, 5, 2),
            (3, 4, 5),
            (10, 10),
            (6, 7),
            (7,)]
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES)
def test_deform_attn_abs(shape, device):
    # Input
    np.random.seed(0)
    torch.manual_seed(42)
    x_ndl = ndl.init.rand(*(shape), device=device, dtype='float32', requires_grad=True)
    x_torch = torch.tensor(x_ndl.cached_data.numpy(), requires_grad=True)

    # Torch forward
    torch_output = torch.abs(x_torch)

    # Local forward
    local_output = ndl.ops.abs(x_ndl)
    np.testing.assert_allclose(torch_output.detach().numpy(), local_output.detach().numpy(), atol=1e-5, rtol=1e-5)

    # Torch backward
    out_grad_torch = torch.randn_like(x_torch)
    out_grad_ndl = ndl.Tensor(out_grad_torch.data.numpy(), device=device, dtype='float32', requires_grad=True)

    torch_output.backward(out_grad_torch, retain_graph=True)
    torch_grad = x_torch.grad.clone()
    x_torch.grad.zero_()

    # Local backward
    local_output.backward(out_grad_ndl)
    ndl_grad = x_ndl.grad

    np.testing.assert_allclose(torch_grad.detach().numpy(), ndl_grad.detach().numpy(), atol=1e-5, rtol=1e-5)

torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64), 
            (2, 32, 64, 64),
            (4, 16, 32, 32),
            (10, 5, 2),
            (3, 4, 5),
            (10, 10),
            (6, 7),
            (7,)]
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES)
def test_deform_attn_sign(shape, device):
    # Input
    np.random.seed(0)
    torch.manual_seed(42)
    x_ndl = ndl.init.rand(*(shape), device=device, dtype='float32', requires_grad=True)
    x_torch = torch.tensor(x_ndl.cached_data.numpy(), requires_grad=True)

    # Torch forward
    torch_output = torch.sign(x_torch)

    # Local forward
    local_output = ndl.ops.sign(x_ndl)
    np.testing.assert_allclose(torch_output.detach().numpy(), local_output.detach().numpy(), atol=1e-5, rtol=1e-5)

    # Torch backward
    out_grad_torch = torch.randn_like(x_torch)
    out_grad_ndl = ndl.Tensor(out_grad_torch.data.numpy(), device=device, dtype='float32', requires_grad=True)

    torch_output.backward(out_grad_torch, retain_graph=True)
    torch_grad = x_torch.grad.clone()
    x_torch.grad.zero_()

    # Local backward
    local_output.backward(out_grad_ndl)
    ndl_grad = x_ndl.grad

    np.testing.assert_allclose(torch_grad.detach().numpy(), ndl_grad.detach().numpy(), atol=1e-5, rtol=1e-5)


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_compare_lucid_our_pos_encoding(conv_qkv_bias, shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        conv_qkv_bias=conv_qkv_bias
    )

    # Ours Deformable Attention Initialization
    our_attn = ndl.nn.DeformableAttention(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        to_q_bias = conv_qkv_bias,
        to_k_bias = conv_qkv_bias,
        to_v_bias = conv_qkv_bias,
        to_out_bias = conv_qkv_bias,
        device=device,
        dtype='float32'
    )

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]

    np.random.seed(0)
    x = ndl.init.rand(*(batch_size, channels, height, width), device=device, dtype='float32', requires_grad=True)
    pytorch_input = torch.tensor(x.cached_data.numpy())

    # (Optional) Copy biases if needed
    if conv_qkv_bias is True:
        lucid_attn.to_q.bias.data = torch.tensor(our_attn.to_q.bias.cached_data.numpy())
        assert lucid_attn.to_q.bias.shape == our_attn.to_q.bias.shape
        assert np.linalg.norm(lucid_attn.to_q.bias.detach().numpy()-our_attn.to_q.bias.detach().numpy()) < 1e-5 

        lucid_attn.to_k.bias.data = torch.tensor(our_attn.to_k.bias.cached_data.numpy())
        assert lucid_attn.to_k.bias.shape == our_attn.to_k.bias.shape
        assert np.linalg.norm(lucid_attn.to_k.bias.detach().numpy()-our_attn.to_k.bias.detach().numpy()) < 1e-5 

        lucid_attn.to_v.bias.data = torch.tensor(our_attn.to_v.bias.cached_data.numpy())
        assert lucid_attn.to_v.bias.shape == our_attn.to_v.bias.shape
        assert np.linalg.norm(lucid_attn.to_v.bias.detach().numpy()-our_attn.to_v.bias.detach().numpy()) < 1e-5 

    # Copy weights
    lucid_attn.to_q.weight.data = torch.tensor(our_attn.to_q.weight.cached_data.numpy())
    assert lucid_attn.to_q.weight.shape == our_attn.to_q.weight.shape
    assert np.linalg.norm(lucid_attn.to_q.weight.detach().numpy()-our_attn.to_q.weight.detach().numpy()) < 1e-5 

    lucid_attn.to_k.weight.data = torch.tensor(our_attn.to_k.weight.cached_data.numpy())
    assert lucid_attn.to_k.weight.shape == our_attn.to_k.weight.shape
    assert np.linalg.norm(lucid_attn.to_k.weight.detach().numpy()-our_attn.to_k.weight.detach().numpy()) < 1e-5 

    lucid_attn.to_v.weight.data = torch.tensor(our_attn.to_v.weight.cached_data.numpy())
    assert lucid_attn.to_v.weight.shape == our_attn.to_v.weight.shape
    assert np.linalg.norm(lucid_attn.to_v.weight.detach().numpy()-our_attn.to_v.weight.detach().numpy()) < 1e-5 

    # Copy offset network weights, for offset network layer0 and layer2 (ConvGp)
    for i in [0, 2]:
        sub_module = lucid_attn.to_offsets[i]
        sub_module.weight.data = torch.tensor(our_attn.to_offsets.modules[i].weight.cached_data.numpy())
        assert sub_module.weight.shape == our_attn.to_offsets.modules[i].weight.shape
        assert np.linalg.norm(sub_module.weight.detach().numpy()-our_attn.to_offsets.modules[i].weight.detach().numpy()) < 1e-5 

        if i == 0:
            sub_module.bias.data = torch.tensor(our_attn.to_offsets.modules[i].bias.cached_data.numpy())
            assert sub_module.bias.shape == our_attn.to_offsets.modules[i].bias.shape
            assert np.linalg.norm(sub_module.bias.detach().numpy()-our_attn.to_offsets.modules[i].bias.detach().numpy()) < 1e-5 

    # Forward pass
    kv_feat_orig, vgrid_scaled_luc, grid_x_luc, grid_x_scaled_luc = lucid_attn(pytorch_input, return_pos_encoding=True)
    our_attn.kv_feats_from_luc = ndl.Tensor(kv_feat_orig.data.numpy(), device=device, dtype='float32', requires_grad=False)
    vgrid_scaled_our, grid_x_our, grid_x_scaled_our = our_attn(x, return_pos_encoding=True)

    # Comapre grid_x
    assert vgrid_scaled_luc.shape == vgrid_scaled_our.shape
    assert np.linalg.norm(vgrid_scaled_luc.detach().numpy()-vgrid_scaled_our.detach().numpy()) < 1e-3 
    assert grid_x_luc.shape == grid_x_our.shape
    assert np.linalg.norm(grid_x_luc.detach().numpy()-grid_x_our.detach().numpy()) < 1e-3 
    assert grid_x_scaled_luc.shape == grid_x_scaled_our.shape
    assert np.linalg.norm(grid_x_scaled_luc.detach().numpy()-grid_x_scaled_our.detach().numpy()) < 1e-3 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_compare_lucid_our_qkv(conv_qkv_bias, shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        conv_qkv_bias=conv_qkv_bias
    )

    # Ours Deformable Attention Initialization
    our_attn = ndl.nn.DeformableAttention(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        to_q_bias = conv_qkv_bias,
        to_k_bias = conv_qkv_bias,
        to_v_bias = conv_qkv_bias,
        to_out_bias = conv_qkv_bias,
        device=device,
        dtype='float32'
    )

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]

    np.random.seed(0)
    x = ndl.init.rand(*(batch_size, channels, height, width), device=device, dtype='float32', requires_grad=True)
    pytorch_input = torch.tensor(x.cached_data.numpy())

    # (Optional) Copy biases if needed
    if conv_qkv_bias is True:
        lucid_attn.to_q.bias.data = torch.tensor(our_attn.to_q.bias.cached_data.numpy())
        assert lucid_attn.to_q.bias.shape == our_attn.to_q.bias.shape
        assert np.linalg.norm(lucid_attn.to_q.bias.detach().numpy()-our_attn.to_q.bias.detach().numpy()) < 1e-5 

        lucid_attn.to_k.bias.data = torch.tensor(our_attn.to_k.bias.cached_data.numpy())
        assert lucid_attn.to_k.bias.shape == our_attn.to_k.bias.shape
        assert np.linalg.norm(lucid_attn.to_k.bias.detach().numpy()-our_attn.to_k.bias.detach().numpy()) < 1e-5 

        lucid_attn.to_v.bias.data = torch.tensor(our_attn.to_v.bias.cached_data.numpy())
        assert lucid_attn.to_v.bias.shape == our_attn.to_v.bias.shape
        assert np.linalg.norm(lucid_attn.to_v.bias.detach().numpy()-our_attn.to_v.bias.detach().numpy()) < 1e-5 

    # Copy weights
    lucid_attn.to_q.weight.data = torch.tensor(our_attn.to_q.weight.cached_data.numpy())
    assert lucid_attn.to_q.weight.shape == our_attn.to_q.weight.shape
    assert np.linalg.norm(lucid_attn.to_q.weight.detach().numpy()-our_attn.to_q.weight.detach().numpy()) < 1e-5 

    lucid_attn.to_k.weight.data = torch.tensor(our_attn.to_k.weight.cached_data.numpy())
    assert lucid_attn.to_k.weight.shape == our_attn.to_k.weight.shape
    assert np.linalg.norm(lucid_attn.to_k.weight.detach().numpy()-our_attn.to_k.weight.detach().numpy()) < 1e-5 

    lucid_attn.to_v.weight.data = torch.tensor(our_attn.to_v.weight.cached_data.numpy())
    assert lucid_attn.to_v.weight.shape == our_attn.to_v.weight.shape
    assert np.linalg.norm(lucid_attn.to_v.weight.detach().numpy()-our_attn.to_v.weight.detach().numpy()) < 1e-5 

    # Copy offset network weights, for offset network layer0 and layer2 (ConvGp)
    for i in [0, 2]:
        sub_module = lucid_attn.to_offsets[i]
        sub_module.weight.data = torch.tensor(our_attn.to_offsets.modules[i].weight.cached_data.numpy())
        assert sub_module.weight.shape == our_attn.to_offsets.modules[i].weight.shape
        assert np.linalg.norm(sub_module.weight.detach().numpy()-our_attn.to_offsets.modules[i].weight.detach().numpy()) < 1e-5 

        if i == 0:
            sub_module.bias.data = torch.tensor(our_attn.to_offsets.modules[i].bias.cached_data.numpy())
            assert sub_module.bias.shape == our_attn.to_offsets.modules[i].bias.shape
            assert np.linalg.norm(sub_module.bias.detach().numpy()-our_attn.to_offsets.modules[i].bias.detach().numpy()) < 1e-5 

    # Forward pass
    kv_feat_orig, kv_feats_luc, k_luc, v_luc, q_luc, sim_luc = lucid_attn(pytorch_input, return_kv_feat=True)
    our_attn.kv_feats_from_luc = ndl.Tensor(kv_feat_orig.data.numpy(), device=device, dtype='float32', requires_grad=False)
    kv_feat_ours, k_our, v_our, q_our, sim_our = our_attn(x, return_kv_feat=True)

    # Comapre kvq
    assert kv_feats_luc.shape == kv_feat_ours.shape
    assert np.linalg.norm(kv_feats_luc.detach().numpy()-kv_feats_luc.detach().numpy()) < 1e-3 

    assert k_luc.shape == k_our.shape
    assert np.linalg.norm(k_luc.detach().numpy()-k_our.detach().numpy()) < 1e-3 

    assert v_luc.shape == v_our.shape
    assert np.linalg.norm(v_luc.detach().numpy()-v_our.detach().numpy()) < 1e-3 

    assert q_luc.shape == q_our.shape
    assert np.linalg.norm(q_luc.detach().numpy()-q_our.detach().numpy()) < 1e-3 

    assert sim_luc.shape == sim_our.shape
    assert np.linalg.norm(sim_luc.detach().numpy()-sim_our.detach().numpy()) < 1e-3 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64),
            (4, 32, 32, 32),
            (8, 32, 64, 64),
            (16, 32, 16, 16)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_compare_lucid_our_normalized_grid(conv_qkv_bias, shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        conv_qkv_bias=conv_qkv_bias
    )

    # Ours Deformable Attention Initialization
    our_attn = ndl.nn.DeformableAttention(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        to_q_bias = conv_qkv_bias,
        to_k_bias = conv_qkv_bias,
        to_v_bias = conv_qkv_bias,
        to_out_bias = conv_qkv_bias,
        device=device,
        dtype='float32'
    )

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]

    np.random.seed(0)
    x = ndl.init.rand(*(batch_size, channels, height, width), device=device, dtype='float32', requires_grad=True)
    pytorch_input = torch.tensor(x.cached_data.numpy())

    # (Optional) Copy biases if needed
    if conv_qkv_bias is True:
        lucid_attn.to_q.bias.data = torch.tensor(our_attn.to_q.bias.cached_data.numpy())
        assert lucid_attn.to_q.bias.shape == our_attn.to_q.bias.shape
        assert np.linalg.norm(lucid_attn.to_q.bias.detach().numpy()-our_attn.to_q.bias.detach().numpy()) < 1e-5 

    # Copy weights
    lucid_attn.to_q.weight.data = torch.tensor(our_attn.to_q.weight.cached_data.numpy())
    assert lucid_attn.to_q.weight.shape == our_attn.to_q.weight.shape
    assert np.linalg.norm(lucid_attn.to_q.weight.detach().numpy()-our_attn.to_q.weight.detach().numpy()) < 1e-5 

    # Copy offset network weights, for offset network layer0 and layer2 (ConvGp)
    for i in [0, 2]:
        sub_module = lucid_attn.to_offsets[i]
        sub_module.weight.data = torch.tensor(our_attn.to_offsets.modules[i].weight.cached_data.numpy())
        assert sub_module.weight.shape == our_attn.to_offsets.modules[i].weight.shape
        assert np.linalg.norm(sub_module.weight.detach().numpy()-our_attn.to_offsets.modules[i].weight.detach().numpy()) < 1e-5 

        if i == 0:
            sub_module.bias.data = torch.tensor(our_attn.to_offsets.modules[i].bias.cached_data.numpy())
            assert sub_module.bias.shape == our_attn.to_offsets.modules[i].bias.shape
            assert np.linalg.norm(sub_module.bias.detach().numpy()-our_attn.to_offsets.modules[i].bias.detach().numpy()) < 1e-5 

    # Forward pass
    lucid_offsets_q, lucid_norm_grid = lucid_attn(pytorch_input, return_norm_vgrid=True)
    output_our, grouped_q_our, offsets_q_our, norm_grid_our = our_attn(x, return_norm_vgrid=True)

    # Comapre offsets
    assert lucid_offsets_q.shape == offsets_q_our.shape
    assert np.linalg.norm(lucid_offsets_q.detach().numpy()-offsets_q_our.detach().numpy()) < 1e-3 

    # Comapre outputs
    assert lucid_norm_grid.shape == norm_grid_our.shape
    assert np.linalg.norm(lucid_norm_grid.detach().numpy()-norm_grid_our.detach().numpy()) < 1e-3 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64),
            (4, 32, 32, 32),
            (8, 32, 64, 64),
            (16, 32, 16, 16)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_compare_lucid_our_offset(conv_qkv_bias, shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        conv_qkv_bias=conv_qkv_bias
    )

    # Ours Deformable Attention Initialization
    our_attn = ndl.nn.DeformableAttention(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        to_q_bias = conv_qkv_bias,
        to_k_bias = conv_qkv_bias,
        to_v_bias = conv_qkv_bias,
        to_out_bias = conv_qkv_bias,
        device=device,
        dtype='float32'
    )

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]

    np.random.seed(0)
    x = ndl.init.rand(*(batch_size, channels, height, width), device=device, dtype='float32', requires_grad=True)
    pytorch_input = torch.tensor(x.cached_data.numpy())

    # (Optional) Copy biases if needed
    if conv_qkv_bias is True:
        lucid_attn.to_q.bias.data = torch.tensor(our_attn.to_q.bias.cached_data.numpy())
        assert lucid_attn.to_q.bias.shape == our_attn.to_q.bias.shape
        assert np.linalg.norm(lucid_attn.to_q.bias.detach().numpy()-our_attn.to_q.bias.detach().numpy()) < 1e-5 

    # Copy weights
    lucid_attn.to_q.weight.data = torch.tensor(our_attn.to_q.weight.cached_data.numpy())
    assert lucid_attn.to_q.weight.shape == our_attn.to_q.weight.shape
    assert np.linalg.norm(lucid_attn.to_q.weight.detach().numpy()-our_attn.to_q.weight.detach().numpy()) < 1e-5 

    # Copy offset network weights, for offset network layer0 and layer2 (ConvGp)
    for i in [0, 2]:
        sub_module = lucid_attn.to_offsets_test[i]
        sub_module.weight.data = torch.tensor(our_attn.to_offsets.modules[i].weight.cached_data.numpy())
        assert sub_module.weight.shape == our_attn.to_offsets.modules[i].weight.shape
        assert np.linalg.norm(sub_module.weight.detach().numpy()-our_attn.to_offsets.modules[i].weight.detach().numpy()) < 1e-5 

        if i == 0:
            sub_module.bias.data = torch.tensor(our_attn.to_offsets.modules[i].bias.cached_data.numpy())
            assert sub_module.bias.shape == our_attn.to_offsets.modules[i].bias.shape
            assert np.linalg.norm(sub_module.bias.detach().numpy()-our_attn.to_offsets.modules[i].bias.detach().numpy()) < 1e-5 

    # Forward pass
    lucid_grouped_q, lucid_offsets_q = lucid_attn(pytorch_input, return_offsets=True)
    output_our, grouped_q_our, offsets_q_our = our_attn(x)

    # Comapre grouped_q
    assert lucid_grouped_q.shape == grouped_q_our.shape
    assert np.linalg.norm(lucid_grouped_q.detach().numpy()-grouped_q_our.detach().numpy()) < 1e-3 

    # Comapre output
    assert lucid_offsets_q.shape == offsets_q_our.shape
    assert np.linalg.norm(lucid_offsets_q.detach().numpy()-offsets_q_our.detach().numpy()) < 1e-3 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_compare_lucid_our_inputq(conv_qkv_bias, shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=6,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        conv_qkv_bias=conv_qkv_bias
    )

    # Ours Deformable Attention Initialization
    our_attn = ndl.nn.DeformableAttention(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=6,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        to_q_bias = conv_qkv_bias,
        to_k_bias = conv_qkv_bias,
        to_v_bias = conv_qkv_bias,
        to_out_bias = conv_qkv_bias,
        device=device,
        dtype='float32'
    )

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]

    np.random.seed(0)
    x = ndl.init.rand(*(batch_size, channels, height, width), device=device, dtype='float32', requires_grad=True)
    pytorch_input = torch.tensor(x.cached_data.numpy())

    # (Optional) Copy biases if needed
    if conv_qkv_bias is True:
        lucid_attn.to_q.bias.data = torch.tensor(our_attn.to_q.bias.cached_data.numpy())
        assert lucid_attn.to_q.bias.shape == our_attn.to_q.bias.shape
        assert np.linalg.norm(lucid_attn.to_q.bias.detach().numpy()-our_attn.to_q.bias.detach().numpy()) < 1e-5 

    # Copy weights
    lucid_attn.to_q.weight.data = torch.tensor(our_attn.to_q.weight.cached_data.numpy())
    assert lucid_attn.to_q.weight.shape == our_attn.to_q.weight.shape
    assert np.linalg.norm(lucid_attn.to_q.weight.detach().numpy()-our_attn.to_q.weight.detach().numpy()) < 1e-5 

    # Forward pass
    output_lucid, lucid_offsets, lucid_orig_q, lucid_orig_x, lucid_grouped_q = lucid_attn(pytorch_input, return_vgrid=True)
    output_our, grouped_q_our, offsets_q_our = our_attn(x, return_vgrid=True)

    # Comapre original q
    assert lucid_orig_q.shape == output_our.shape
    assert np.linalg.norm(lucid_orig_q.detach().numpy()-output_our.detach().numpy()) < 1e-3 

    # Comapre grouped_q
    assert lucid_grouped_q.shape == grouped_q_our.shape
    assert np.linalg.norm(lucid_grouped_q.detach().numpy()-grouped_q_our.detach().numpy()) < 1e-3 

torch.manual_seed(42)
conv_shapes = [
    (1, 1, 1, 3),
    (1, 1, 2, 3),
    (1, 1, 4, 3),
    (1, 1, 4, 3)
]
x_shapes = [(1, 32, 64, 64), 
            (2, 32, 64, 64),
            (4, 16, 32, 32)]
conv_qkv_bias = [False, True]
out_channels = [16, 32, 64]
@pytest.mark.parametrize("out_channel", out_channels)
@pytest.mark.parametrize("stride, padding, groups, kernel_size", conv_shapes)
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN)
def test_deform_attn_group_conv(out_channel, stride, padding, groups, kernel_size, conv_qkv_bias, shape, device):
    bs, in_channels, height, width = shape
    # Needle setup
    needle_conv = ndl.nn.ConvGp(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, groups=groups, bias=conv_qkv_bias, device=device, dtype='float32')
    #needle_conv = ndl.nn.Conv(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, bias=conv_qkv_bias, device=device, dtype='float32')

    # PyTorch setup
    pytorch_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=conv_qkv_bias)

    #pytorch_conv.weight.data = torch.tensor(needle_conv.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    pytorch_conv.weight.data = torch.tensor(needle_conv.weight.cached_data.numpy())
    if conv_qkv_bias:
        pytorch_conv.bias.data = torch.tensor(needle_conv.bias.cached_data.numpy())

    # Input
    np.random.seed(0)
    x = ndl.init.rand(*(bs, in_channels, height, width), device=device, dtype='float32', requires_grad=True)
    pytorch_input = torch.tensor(x.cached_data.numpy())

    # Forward pass
    pytorch_output = pytorch_conv(pytorch_input)  # NCHW
    needle_output = needle_conv(x)

    # Comparison
    assert pytorch_conv.weight.shape == needle_conv.weight.shape
    assert np.linalg.norm(pytorch_conv.weight.detach().numpy()-needle_conv.weight.detach().numpy()) < 1e-5 
    if conv_qkv_bias:
        assert np.linalg.norm(pytorch_conv.bias.detach().numpy()-needle_conv.bias.detach().numpy()) < 1e-5 
        assert pytorch_conv.bias.shape == needle_conv.bias.shape

    assert needle_output.shape == pytorch_output.shape
    assert np.linalg.norm(needle_output.cached_data.numpy() - pytorch_output.data.numpy()) < 1e-3


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_compare_inputq(conv_qkv_bias, shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=4,                    # Dimension per head
        heads=8,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=6,           # Offset kernel size
        group_queries=True,
        group_key_values=False,
        conv_qkv_bias=conv_qkv_bias
    )    

    # DAT Deformable Attention Initialization
    dat_attn = DAttentionBaseline(
        q_size=(64, 64),               # Query size (H, W)
        kv_size=(64, 64),              # Key/Value size (H, W)
        n_heads=8,                     # Number of attention heads
        n_head_channels=4,             # Channels per head (Total dim = n_heads * n_head_channels = 32)
        n_groups=2,                    # No grouping
        attn_drop=0.,                  # Attention dropout
        proj_drop=0.,                  # Projection dropout
        stride=4,                      # Downsample factor
        offset_range_factor=4,         # Offset scale
        use_pe=True,                  # No positional encoding
        dwc_pe=False,                  # No depthwise convolution positional encoding
        no_off=False,                  # Enable offsets
        fixed_pe=False,                # No fixed positional encoding
        ksize=6,                       # Offset kernel size
        log_cpb=False,                  # No logarithmic CPB
        conv_qkv_bias=conv_qkv_bias
    )    

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass through DAT Deformable Attention
    output_dat, dat_offsets, dat_ref, dat_orig_q, dat_orig_x = dat_attn(x)    

    # Copy the weights from source_model to target_model
    lucid_attn.to_q.weight.data = dat_attn.proj_q.weight.clone()

    # (Optional) Copy biases if needed
    if dat_attn.conv_qkv_bias is True:
        lucid_attn.conv_qkv_bias = dat_attn.conv_qkv_bias    
        lucid_attn.to_q.bias.data = dat_attn.proj_q.bias.clone()    
        #print(f"lucid_attn.to_q.bias.shape = {lucid_attn.to_q.bias.shape}")
        #print(f"dat_attn.proj_q.bias.shape = {dat_attn.proj_q.bias.shape}")
        assert torch.allclose(dat_attn.proj_q.bias, lucid_attn.to_q.bias, atol=1e-5)
        assert np.linalg.norm(dat_attn.proj_q.bias.detach().numpy()-lucid_attn.to_q.bias.detach().numpy()) < 1e-5 

    # Forward pass through Lucidrains' Deformable Attention
    dat_to_q_weight = dat_attn.proj_q.weight.data
    lucid_to_q_weight = lucid_attn.to_q.weight.data
    #print(f"dat_to_q_weight.shape = {dat_to_q_weight.shape}")
    #print(f"lucid_to_q_weight.shape = {lucid_to_q_weight.shape}")
    output_lucid, lucid_offsets, lucid_orig_q, lucid_orig_x, lucid_grouped_q = lucid_attn(x, return_vgrid=True)

    # Compare projection weight of q
    assert dat_to_q_weight.shape == lucid_to_q_weight.shape
    assert torch.allclose(dat_to_q_weight, lucid_to_q_weight, atol=1e-5)
    assert np.linalg.norm(dat_to_q_weight.detach().numpy()-lucid_to_q_weight.detach().numpy()) < 1e-5 

    # Comapre original q
    assert lucid_orig_q.shape == dat_orig_q.shape
    assert np.linalg.norm(lucid_orig_q.detach().numpy()-dat_orig_q.detach().numpy()) < 1e-5 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_compare_inputx(shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
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

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]
    x = torch.randn(batch_size, channels, height, width)

    # Forward pass through DAT Deformable Attention
    output_dat, dat_offsets, dat_ref, dat_orig_q, dat_orig_x = dat_attn(x)    

    # Forward pass through Lucidrains' Deformable Attention
    output_lucid, lucid_offsets, lucid_orig_q, lucid_orig_x, lucid_grouped_q = lucid_attn(x, return_vgrid=True)

    # Comapre original q
    assert lucid_orig_x.shape == dat_orig_x.shape
    assert np.linalg.norm(lucid_orig_x.detach().numpy()-dat_orig_x.detach().numpy()) < 1e-5 


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
