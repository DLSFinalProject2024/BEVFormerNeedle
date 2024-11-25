import sys

import psutil
import os
sys.path.append("./python")
import numpy as np
import pytest
import mugrade
import torch
import torch.nn.functional as F
import needle as ndl
from needle import ops
from needle import backend_ndarray as nd

from deformable_attention import DeformableAttention
from deformable_attention import DeformableAttention1D


_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")
    ),
]

_DEVICES_ATTN = [
    nd.cpu()
]
def print_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage = mem_info.rss / (1024 ** 2)
    print(f"[{message}] Memory usage: {mem_usage:.2f} MB")  # RSS: Resident Set Size
    return mem_usage

import threading
import time

def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    while True:
        mem_info = process.memory_info()
        print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")
        time.sleep(0.05)  # Adjust the sleep interval as needed

'''
def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides


def check_same_memory(original, view):
    assert original._handle.ptr() == view._handle.ptr()


""" For converting slice notation to slice objects to make some proceeding tests easier to read """


class _ShapeAndSlices(nd.NDArray):
    def __getitem__(self, idxs):
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        return self.shape, idxs


ShapeAndSlices = lambda *shape: _ShapeAndSlices(np.ones(shape))


@pytest.mark.parametrize(
    "params",
    [
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:2, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:3, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6],
            "rhs": ShapeAndSlices(7, 7, 7)[:2, :3, :4],
        },
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_ewise(params, device):
    lhs_shape, lhs_slices = params["lhs"]
    rhs_shape, rhs_slices = params["rhs"]
    _A = np.random.randn(*lhs_shape)
    _B = np.random.randn(*rhs_shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    start_ptr = A._handle.ptr()
    A[lhs_slices] = B[rhs_slices]
    _A[lhs_slices] = _B[rhs_slices]
    end_ptr = A._handle.ptr()
    assert start_ptr == end_ptr, "you should modify in-place"
    compare_strides(_A, A)
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)


# Ex: We want arrays of size (4, 5, 6) setting element(s) [1:4, 2, 3] to a scalar
@pytest.mark.parametrize(
    "params",
    [
        ShapeAndSlices(4, 5, 6)[1, 2, 3],
        ShapeAndSlices(4, 5, 6)[1:4, 2, 3],
        ShapeAndSlices(4, 5, 6)[:4, 2:5, 3],
        ShapeAndSlices(4, 5, 6)[1::2, 2:5, ::2],
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_scalar(params, device):
    shape, slices = params
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    # probably tear these out using lambdas
    # print(slices)
    start_ptr = A._handle.ptr()
    _A[slices] = 4.0
    A[slices] = 4.0
    end_ptr = A._handle.ptr()
    assert start_ptr == end_ptr, "you should modify in-place"
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
    compare_strides(_A, A)


OPS = {
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "equal": lambda a, b: a == b,
    "greater_than": lambda a, b: a >= b,
}
OP_FNS = [OPS[k] for k in OPS]
OP_NAMES = [k for k in OPS]

ewise_shapes = [(1, 1, 1), (4, 5, 6)]

@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn_my(fn, shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)
'''


torch.manual_seed(42)
x_shapes = [(1, 32, 64, 64)]
#x_shapes = [(1, 512, 64, 64)] #will be Killed
@pytest.mark.parametrize("shape", x_shapes)
@pytest.mark.parametrize("device", _DEVICES_ATTN, ids=["cpu"])
def test_deform_attn_lucidrains(shape, device):
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
def test_deform_attn_1D_lucidrains(shape):
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
    (1, 1, 1, 1),
    (1, 1, 3, 3),
    (1, 1, 3, 4),
    (1, 1, 4, 3),
    (1, 3, 5, 100),
    (1, 3, 32, 16),
    (2, 3, 32, 32),
    (4, 1, 64, 64),
    (8, 3, 128, 128),
]

# @pytest.mark.parametrize("N,C,H,W", grid_sample_params)
# @pytest.mark.parametrize("mode", ['bilinear', 'nearest'])
# @pytest.mark.parametrize("padding_mode", ['zeros', 'border', 'reflection'])
# @pytest.mark.parametrize("align_corners", [True, False])
# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("N,C,H,W", grid_sample_params)
@pytest.mark.parametrize("mode", ['bilinear'])
@pytest.mark.parametrize("padding_mode", ['zeros'])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_grid_sample(N, C, H, W, mode, padding_mode, align_corners, device):
    np.random.seed(0)
    a = ndl.init.rand(N, C, H, W, device=device)
    grid = ndl.init.rand(N, H, W, 2, low=-1.0, high=1.0, device=device)
    out = ops.grid_sample(a, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    a_torch = torch.tensor(a.cached_data.numpy(), requires_grad=False)
    grid_torch = torch.tensor(grid.cached_data.numpy(), requires_grad=False)
    out_torch = torch.nn.functional.grid_sample(
        a_torch, grid_torch, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    out_np = out.cached_data.numpy()
    out_torch_np = out_torch.detach().cpu().numpy()
    assert np.linalg.norm(out_np - out_torch_np) < 1e-3

if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")
