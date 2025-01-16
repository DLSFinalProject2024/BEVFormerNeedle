import sys

import psutil
import os
sys.path.append("./python")
import numpy as np
import pytest
import needle as ndl
from needle import backend_ndarray as nd
from needle import ops
import threading
import time

import torch
import torch.nn
import torch.nn.functional as F
# Lucidrians-Deformable Attention
sys.path.append('./LUC')
from deformable_attention import DeformableAttention
from deformable_attention import DeformableAttention1D
from deformable_attention import DeformableAttention2D
from deformable_attention_2d_local import DeformableAttention2DLocal


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


bias_bool = [False, True]
@pytest.mark.parametrize("in_features, out_features, batch_size", [
    (2, 4, 2),
    (10, 20, 5)
])
@pytest.mark.parametrize("bias_bool", bias_bool)
@pytest.mark.parametrize("device", _DEVICES)
def test_linear_forward(in_features, out_features, batch_size, device, bias_bool):
    # Initialize random input
    channel = 4096
    d_feature = 256
    shape = (batch_size, channel, d_feature, in_features)
    input_ndl = ndl.init.rand(*(shape), device=device, dtype='float32', requires_grad=True)
    input_torch = torch.tensor(input_ndl.cached_data.numpy(), requires_grad=True)

    # Initialize PyTorch's Linear layer
    torch_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias_bool)
    
    # Initialize your custom Linear layer
    my_linear = ndl.nn.Linear(in_features=in_features, out_features=out_features, bias=bias_bool, device=device, dtype='float32')
    
    # Copy weights and biases from torch Linear to your Linear for fair comparison
    if bias_bool:
        torch_linear.bias.data = torch.tensor(my_linear.bias.cached_data.numpy())
    torch_linear.weight.data = torch.tensor(my_linear.weight.cached_data.numpy().T)

    # Perform forward pass
    torch_output = torch_linear(input_torch)
    input_ndl = input_ndl.reshape((batch_size*channel*d_feature, in_features))
    my_output = my_linear(input_ndl)
    my_output = my_output.reshape((batch_size, channel, d_feature, out_features))

    # Check if outputs are close
    np.testing.assert_allclose(my_output.detach().numpy(), torch_output.detach().numpy(), atol=1e-5, rtol=1e-5)

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
def test_deform_attn_gelu(shape, device):
    # Input
    np.random.seed(0)
    torch.manual_seed(42)
    x_ndl = ndl.init.rand(*(shape), device=device, dtype='float32', requires_grad=True)
    x_torch = torch.tensor(x_ndl.cached_data.numpy(), requires_grad=True)

    # Torch forward
    torch_output = torch.nn.GELU(approximate='tanh')(x_torch)

    # Local forward
    local_output = ndl.nn.GELU()(x_ndl)
    np.testing.assert_allclose(torch_output.detach().numpy(), local_output.detach().numpy(), atol=1e-5, rtol=1e-5)

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

def copy_weights_and_biase_block(source_cpb_block, target_cpb_block, bias=True, plain_module=False):
    """
    Copies weights and biases from source_cpb to target_cpb.
    
    Args:
        source_cpb: The original CPB object with weights to copy.
        target_cpb: The target CPB object to which weights will be copied.
    """
    # Get all Linear layers from source and target
    source_linear_layers = []
    target_linear_layers = []

    # Extract Linear layers from source CPB
    if not plain_module:
        for module in source_cpb_block.modules:
            if isinstance(module, ndl.nn.Sequential):
                for layer in module.modules:
                    if isinstance(layer, ndl.nn.Linear):
                        source_linear_layers.append(layer)
            elif isinstance(module, ndl.nn.Linear):  # Handle final Linear layer
                source_linear_layers.append(module)
    else:
        if isinstance(source_cpb_block, ndl.nn.Linear):
            module = source_cpb_block
            if isinstance(module, ndl.nn.Sequential):
                for layer in module.modules:
                    if isinstance(layer, ndl.nn.Linear):
                        source_linear_layers.append(layer)
            elif isinstance(module, ndl.nn.Linear):  # Handle final Linear layer
                source_linear_layers.append(module)


    # Extract Linear layers from target CPB
    if not plain_module:
        for module in target_cpb_block:
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.Linear):
                        target_linear_layers.append(layer)
            elif isinstance(module, torch.nn.Linear):  # Handle final Linear layer
                target_linear_layers.append(module)
    else:
        if isinstance(target_cpb_block, torch.nn.Linear):
            module = target_cpb_block
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.Linear):
                        target_linear_layers.append(layer)
            elif isinstance(module, torch.nn.Linear):  # Handle final Linear layer
                target_linear_layers.append(module)


    # Ensure the number of Linear layers match
    assert len(source_linear_layers) == len(target_linear_layers), \
        "Number of Linear layers in source and target CPBs do not match!"

    # Copy weights and biases
    for src_layer, tgt_layer in zip(source_linear_layers, target_linear_layers):
        if bias:
            tgt_layer.bias.data = torch.tensor(src_layer.bias.cached_data.numpy())
        tgt_layer.weight.data = torch.tensor(src_layer.weight.cached_data.numpy().T)

        if bias:
            assert tgt_layer.bias.shape == src_layer.bias.shape
            assert np.linalg.norm(tgt_layer.bias.detach().numpy()-src_layer.bias.detach().numpy()) < 1e-5 

        assert tgt_layer.weight.shape == src_layer.weight.transpose((0, 1)).shape
        assert np.linalg.norm(tgt_layer.weight.detach().numpy()-src_layer.weight.detach().numpy().T) < 1e-5 


def copy_weights_and_biases(source_cpb, target_cpb, bias=True, plain_module=False):
    """
    Copies weights and biases from source_cpb to target_cpb.
    
    Args:
        source_cpb: The original CPB object with weights to copy.
        target_cpb: The target CPB object to which weights will be copied.
    """
    # Get all Linear layers from source and target
    source_linear_layers = []
    target_linear_layers = []

    # Extract Linear layers from source CPB
    if not plain_module:
        for module in source_cpb.cpb_block.modules:
            if isinstance(module, ndl.nn.Sequential):
                for layer in module.modules:
                    if isinstance(layer, ndl.nn.Linear):
                        source_linear_layers.append(layer)
            elif isinstance(module, ndl.nn.Linear):  # Handle final Linear layer
                source_linear_layers.append(module)
    else:
        if isinstance(source_cpb.cpb_block, ndl.nn.Linear):
            module = source_cpb.cpb_block
            if isinstance(module, ndl.nn.Sequential):
                for layer in module.modules:
                    if isinstance(layer, ndl.nn.Linear):
                        source_linear_layers.append(layer)
            elif isinstance(module, ndl.nn.Linear):  # Handle final Linear layer
                source_linear_layers.append(module)


    # Extract Linear layers from target CPB
    if not plain_module:
        for module in target_cpb.mlp:
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.Linear):
                        target_linear_layers.append(layer)
            elif isinstance(module, torch.nn.Linear):  # Handle final Linear layer
                target_linear_layers.append(module)
    else:
        if isinstance(target_cpb.mlp, torch.nn.Linear):
            module = target_cpb.mlp
            if isinstance(module, torch.nn.Sequential):
                for layer in module:
                    if isinstance(layer, torch.nn.Linear):
                        target_linear_layers.append(layer)
            elif isinstance(module, torch.nn.Linear):  # Handle final Linear layer
                target_linear_layers.append(module)


    # Ensure the number of Linear layers match
    assert len(source_linear_layers) == len(target_linear_layers), \
        "Number of Linear layers in source and target CPBs do not match!"

    # Copy weights and biases
    for src_layer, tgt_layer in zip(source_linear_layers, target_linear_layers):
        if bias:
            tgt_layer.bias.data = torch.tensor(src_layer.bias.cached_data.numpy())
        tgt_layer.weight.data = torch.tensor(src_layer.weight.cached_data.numpy().T)

        if bias:
            assert tgt_layer.bias.shape == src_layer.bias.shape
            assert np.linalg.norm(tgt_layer.bias.detach().numpy()-src_layer.bias.detach().numpy()) < 1e-5 

        assert tgt_layer.weight.shape == src_layer.weight.transpose((0, 1)).shape
        assert np.linalg.norm(tgt_layer.weight.detach().numpy()-src_layer.weight.detach().numpy().T) < 1e-5 

torch.manual_seed(42)
np.random.seed(42)
x_shapes = [(64, 3, 32, 32)]
conv_qkv_bias = [False, True]
conv_out_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("conv_out_bias", conv_out_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
#@pytest.mark.parametrize("device", _DEVICES)
def test_deform_attn_compare_lucid_our_attn_cifar10(conv_out_bias, conv_qkv_bias, shape, device):
    # Launch a new thread to monitor the memory usage
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()

    # Lucidrains Deformable Attention Initialization
    lucid_attn = DeformableAttention2DLocal(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=8,                    # Dimension per head
        heads=4,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=True,
        conv_qkv_bias=conv_qkv_bias,
        conv_out_bias=conv_out_bias
    )

    # Ours Deformable Attention Initialization
    our_attn = ndl.nn.DeformableAttention(
        dim=32,                        # Feature dimensions (C = 32)
        dim_head=8,                    # Dimension per head
        heads=4,                       # Attention heads
        dropout=0.,                    # Dropout
        downsample_factor=4,           # Downsample factor
        offset_scale=4,                # Offset scale
        offset_groups=2,              # No offset groups
        offset_kernel_size=5,           # Offset kernel size
        group_queries=True,
        group_key_values=True,
        to_q_bias = conv_qkv_bias,
        to_k_bias = conv_qkv_bias,
        to_v_bias = conv_qkv_bias,
        to_out_bias = conv_out_bias,
        device=device,
        dtype='float32'
    )
    conv_preprocess = ndl.nn.ConvGp(3, 32, 3, groups = 1, bias=True, device=device, dtype='float32')

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]

    x = ndl.init.rand(*(batch_size, channels, height, width), device=device, dtype='float32', requires_grad=True)
    x = conv_preprocess(x) #(B, C=32, H, W)
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

    if conv_out_bias is True:
        lucid_attn.to_out.bias.data = torch.tensor(our_attn.to_out.bias.cached_data.numpy())
        assert lucid_attn.to_out.bias.shape == our_attn.to_out.bias.shape
        assert np.linalg.norm(lucid_attn.to_out.bias.detach().numpy()-our_attn.to_out.bias.detach().numpy()) < 1e-5 

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

    lucid_attn.to_out.weight.data = torch.tensor(our_attn.to_out.weight.cached_data.numpy())
    assert lucid_attn.to_out.weight.shape == our_attn.to_out.weight.shape
    assert np.linalg.norm(lucid_attn.to_out.weight.detach().numpy()-our_attn.to_out.weight.detach().numpy()) < 1e-5 

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

    # Copy weights and bias of Linear() in CPB
    copy_weights_and_biases(our_attn.rel_pos_bias, lucid_attn.rel_pos_bias)

    # Forward pass
    kv_feat_orig, sim_luc, attn_luc, out_luc = lucid_attn(pytorch_input, return_attn=True)
    #our_attn.kv_feats_from_luc = ndl.Tensor(kv_feat_orig.data.numpy(), device=device, dtype='float32', requires_grad=False)
    sim_our, attn_our, out_our = our_attn(x, return_attn=True)

    # Comapre
    assert attn_luc.shape == attn_our.shape
    assert np.linalg.norm(attn_luc.detach().numpy()-attn_our.detach().numpy()) < 1e-3 
    assert out_luc.shape == out_our.shape
    assert np.linalg.norm(out_luc.detach().numpy()-out_our.detach().numpy()) < 2e-3 


torch.manual_seed(42)
np.random.seed(42)
x_shapes = [(1, 32, 64, 64),
            (4, 32, 32, 32),
            (8, 32, 16, 16)]
conv_qkv_bias = [False, True]
conv_out_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("conv_out_bias", conv_out_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
#@pytest.mark.parametrize("device", _DEVICES)
def test_deform_attn_compare_lucid_our_attn(conv_out_bias, conv_qkv_bias, shape, device):
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
        conv_qkv_bias=conv_qkv_bias,
        conv_out_bias=conv_out_bias
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
        to_out_bias = conv_out_bias,
        device=device,
        dtype='float32'
    )

    batch_size = shape[0]
    channels   = shape[1]
    height     = shape[2]
    width      = shape[3]

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

    if conv_out_bias is True:
        lucid_attn.to_out.bias.data = torch.tensor(our_attn.to_out.bias.cached_data.numpy())
        assert lucid_attn.to_out.bias.shape == our_attn.to_out.bias.shape
        assert np.linalg.norm(lucid_attn.to_out.bias.detach().numpy()-our_attn.to_out.bias.detach().numpy()) < 1e-5 

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

    lucid_attn.to_out.weight.data = torch.tensor(our_attn.to_out.weight.cached_data.numpy())
    assert lucid_attn.to_out.weight.shape == our_attn.to_out.weight.shape
    assert np.linalg.norm(lucid_attn.to_out.weight.detach().numpy()-our_attn.to_out.weight.detach().numpy()) < 1e-5 

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

    # Copy weights and bias of Linear() in CPB
    copy_weights_and_biases(our_attn.rel_pos_bias, lucid_attn.rel_pos_bias)

    # Forward pass
    kv_feat_orig, sim_luc, attn_luc, out_luc = lucid_attn(pytorch_input, return_attn=True)
    #our_attn.kv_feats_from_luc = ndl.Tensor(kv_feat_orig.data.numpy(), device=device, dtype='float32', requires_grad=False)
    sim_our, attn_our, out_our = our_attn(x, return_attn=True)

    # Comapre
    assert sim_luc.shape == sim_our.shape
    assert np.linalg.norm(sim_luc.detach().numpy()-sim_our.detach().numpy()) < 1e-2 
    assert attn_luc.shape == attn_our.shape
    assert np.linalg.norm(attn_luc.detach().numpy()-attn_our.detach().numpy()) < 1e-3 
    assert out_luc.shape == out_our.shape
    assert np.linalg.norm(out_luc.detach().numpy()-out_our.detach().numpy()) < 1e-3 


torch.manual_seed(42)
#x_shapes = [(1, 32, 64, 64)]
x_shapes = [(1, 32, 64, 64),
            (4, 32, 32, 32),
            (8, 32, 32, 32),
            (16, 32, 16, 16)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
#@pytest.mark.parametrize("device", _DEVICES)
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

    # Copy weights and bias of Linear() in CPB
    copy_weights_and_biases(our_attn.rel_pos_bias, lucid_attn.rel_pos_bias)
    #copy_weights_and_biase_block(our_attn.rel_pos_bias.cpb_block2, lucid_attn.rel_pos_bias.mlp2, bias=False, plain_module=True)

    # Forward pass
    kv_feat_orig, vgrid_scaled_luc, grid_x_luc, grid_x_scaled_luc, rel_pos_bias_luc, sim_luc, pos_back_luc, bias_back_luc, bias_to_luc, bias_from_luc = lucid_attn(pytorch_input, return_pos_encoding=True)
    #kv_feat_orig, pos_back_luc, bias_back_luc, bias_to_luc, bias_from_luc = lucid_attn(pytorch_input, return_pos_encoding=True, return_bias_only=True)
    #our_attn.kv_feats_from_luc = ndl.Tensor(kv_feat_orig.data.numpy(), device=device, dtype='float32', requires_grad=False)
    vgrid_scaled_our, grid_x_our, grid_x_scaled_our, rel_pos_bias_our, sim_our, pos_back_our, bias_back_our, bias_to_our, bias_from_our = our_attn(x, return_pos_encoding=True)
    #pos_back_our, bias_back_our, bias_to_our, bias_from_our = our_attn(x, return_pos_encoding=True, return_bias_only=True)

    # Comapre
    assert vgrid_scaled_luc.shape == vgrid_scaled_our.shape
    assert np.linalg.norm(vgrid_scaled_luc.detach().numpy()-vgrid_scaled_our.detach().numpy()) < 1e-3 
    assert grid_x_luc.shape == grid_x_our.shape
    assert np.linalg.norm(grid_x_luc.detach().numpy()-grid_x_our.detach().numpy()) < 1e-3 
    assert grid_x_scaled_luc.shape == grid_x_scaled_our.shape
    assert np.linalg.norm(grid_x_scaled_luc.detach().numpy()-grid_x_scaled_our.detach().numpy()) < 1e-3 

    assert pos_back_luc.shape == pos_back_our.shape
    assert np.linalg.norm(pos_back_luc.detach().numpy()-pos_back_our.detach().numpy()) < 1e-3 
    assert bias_back_luc.shape == bias_back_our.shape
    assert np.linalg.norm(bias_back_luc.detach().numpy()-bias_back_our.detach().numpy()) < 1e-3 
    assert bias_to_luc.shape == bias_to_our.shape
    assert np.linalg.norm(bias_to_luc.detach().numpy()-bias_to_our.detach().numpy()) < 1e-3 
    assert bias_from_luc.shape == bias_from_our.shape
    assert np.linalg.norm(bias_from_luc.detach().numpy()-bias_from_our.detach().numpy()) < 1e-3 

    assert rel_pos_bias_luc.shape == rel_pos_bias_our.shape
    assert np.linalg.norm(rel_pos_bias_luc.detach().numpy()-rel_pos_bias_our.detach().numpy()) < 1e-3 
    assert sim_luc.shape == sim_our.shape
    assert np.linalg.norm(sim_luc.detach().numpy()-sim_our.detach().numpy()) < 1e-2 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
#@pytest.mark.parametrize("device", _DEVICES)
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
    kv_feat_orig, group_x_luc, vgrid_scaled_luc, kv_feats_luc, k_luc, v_luc, q_luc, sim_luc = lucid_attn(pytorch_input, return_kv_feat=True)
    #our_attn.kv_feats_orig = ndl.Tensor(kv_feat_orig.data.numpy(), device=device, dtype='float32', requires_grad=False)
    kv_feat_ours, group_x_our, vgrid_scaled_our, k_our, v_our, q_our, sim_our = our_attn(x, return_kv_feat=True)

    # Comapre kvq
    assert kv_feat_orig.shape == our_attn.kv_feats_orig.shape
    assert np.linalg.norm(kv_feat_orig.detach().numpy()-our_attn.kv_feats_orig.detach().numpy()) < 1e-4

    assert group_x_luc.shape == group_x_our.shape
    assert np.linalg.norm(group_x_luc.detach().numpy()-group_x_our.detach().numpy()) < 1e-4

    assert vgrid_scaled_luc.shape == vgrid_scaled_our.shape
    assert np.linalg.norm(vgrid_scaled_luc.detach().numpy()-vgrid_scaled_our.detach().numpy()) < 1e-4

    assert kv_feats_luc.shape == kv_feat_ours.shape
    assert np.linalg.norm(kv_feats_luc.detach().numpy()-kv_feats_luc.detach().numpy()) < 1e-3 

    assert k_luc.shape == k_our.shape
    assert np.linalg.norm(k_luc.detach().numpy()-k_our.detach().numpy()) < 1e-3 

    assert v_luc.shape == v_our.shape
    assert np.linalg.norm(v_luc.detach().numpy()-v_our.detach().numpy()) < 1e-3 

    assert q_luc.shape == q_our.shape
    assert np.linalg.norm(q_luc.detach().numpy()-q_our.detach().numpy()) < 1e-3 

    assert sim_luc.shape == sim_our.shape
    assert np.linalg.norm(sim_luc.detach().numpy()-sim_our.detach().numpy()) < 1e-2 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64),
            (4, 32, 32, 32),
            (8, 32, 32, 32),
            (16, 32, 16, 16)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
#@pytest.mark.parametrize("device", _DEVICES)
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
    offsets_q_our, norm_grid_our = our_attn(x, return_norm_vgrid=True)

    # Comapre offsets
    assert lucid_offsets_q.shape == offsets_q_our.shape
    assert np.linalg.norm(lucid_offsets_q.detach().numpy()-offsets_q_our.detach().numpy()) < 1e-3 

    # Comapre outputs
    assert lucid_norm_grid.shape == norm_grid_our.shape
    assert np.linalg.norm(lucid_norm_grid.detach().numpy()-norm_grid_our.detach().numpy()) < 1e-3 


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64),
            (4, 32, 32, 32),
            (8, 32, 32, 32),
            (16, 32, 16, 16)]
conv_qkv_bias = [False, True]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("conv_qkv_bias", conv_qkv_bias)
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
#@pytest.mark.parametrize("device", _DEVICES)
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
    grouped_q_our, offsets_q_our = our_attn(x, return_offsets=True)

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
#@pytest.mark.parametrize("device", _DEVICES)
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
    lucid_orig_q, lucid_grouped_q = lucid_attn(pytorch_input, return_orig_q=True)
    orig_q_our, grouped_q_our = our_attn(x, return_orig_q=True)

    # Comapre original q
    assert lucid_orig_q.shape == orig_q_our.shape
    assert np.linalg.norm(lucid_orig_q.detach().numpy()-orig_q_our.detach().numpy()) < 1e-3 

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
#@pytest.mark.parametrize("device", _DEVICES_ATTN)
@pytest.mark.parametrize("device", _DEVICES)
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

# Parameters for testing grid_sample
grid_sample_params = [
    (1, 1, 1, 1, 1, 2),
    (1, 1, 3, 3, 3, 3),
    (1, 1, 3, 4, 3, 4),
    (1, 2, 4, 3, 4, 3),
    (1, 3, 5, 100, 16, 16),
    (1, 3, 32, 16, 1, 1),
    (2, 3, 32, 32, 15, 18),
    (4, 1, 64, 64, 128, 1),
]

# @pytest.mark.parametrize("N,C,H,W,H_out,W_out", grid_sample_params)
# @pytest.mark.parametrize("mode", ['bilinear', 'nearest'])
# @pytest.mark.parametrize("padding_mode", ['zeros', 'border', 'reflection'])
# @pytest.mark.parametrize("align_corners", [True, False])
# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("N,C,H,W,H_out,W_out", grid_sample_params)
@pytest.mark.parametrize("mode", ['bilinear'])
@pytest.mark.parametrize("padding_mode", ['zeros'])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("device", _DEVICES_ATTN)
def test_nn_grid_sample(N, C, H, W, H_out, W_out, mode, padding_mode, align_corners, device):
    np.random.seed(0)
    a = ndl.init.rand(N, C, H, W, device=device, requires_grad=True)
    grid = ndl.init.rand(N, H_out, W_out, 2, low=-1.0, high=1.0, device=device, requires_grad=True)
    out = ops.grid_sample(a, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    a_torch = torch.tensor(a.cached_data.numpy(), requires_grad=True)
    grid_torch = torch.tensor(grid.cached_data.numpy(), requires_grad=True)
    out_torch = torch.nn.functional.grid_sample(
        a_torch, grid_torch, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    out_np = out.cached_data.numpy()
    out_torch_np = out_torch.detach().cpu().numpy()
    assert np.linalg.norm(out_np - out_torch_np) < 1e-3

    out.sum().backward()
    out_torch.sum().backward()
    assert np.linalg.norm(a_torch.grad.data.numpy() - a.grad.cached_data.numpy()) < 1e-3
    assert np.linalg.norm(grid_torch.grad.data.numpy() - grid.grad.cached_data.numpy()) < 1e-3


if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")
