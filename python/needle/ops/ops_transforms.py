from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

class GridSample(TensorOp):
    def __init__(self, mode: str, padding_mode: str, align_corners: bool):
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        assert mode in ['bilinear', 'nearest', 'bicubic']
        assert padding_mode in ['zeros', 'border', 'reflection']
    def compute(self, a: NDArray, grid: NDArray):
        return a.grid_sample(grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
    def gradient(self, out_grad: Tensor, node: Tensor):
        a, grid = node.inputs
        return tuple(grid_sample_backward(out_grad, a, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners))

def grid_sample(a, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    return GridSample(mode, padding_mode, align_corners)(a, grid)


class GridSampleBackward(TensorTupleOp):
    def __init__(self, mode: str, padding_mode: str, align_corners: bool):
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        assert mode in ['bilinear', 'nearest', 'bicubic']
        assert padding_mode in ['zeros', 'border', 'reflection']
    def compute(self, out_grad: NDArray, a: NDArray, grid: NDArray):
        return out_grad.grid_sample_backward(a, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
    def gradient(self, out_grad: Tensor, node: Tensor):
        raise NotImplementedError

def grid_sample_backward(out_grad, a, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    return GridSampleBackward(mode, padding_mode, align_corners)(out_grad, a, grid)