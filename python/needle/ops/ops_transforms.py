from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

class GridSample:
    def __init__(self, mode: str, padding_mode: str, align_corners: bool):
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        assert mode in ['bilinear', 'nearest', 'bicubic']
        assert padding_mode in ['zeros', 'border', 'reflection']
    def compute(self, a: NDArray, grid: NDArray):
        return a.grid_sample(grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
    def gradient(self, out_grad: Tensor, node: Tensor):
        pass
def grid_sample(a, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    return GridSample(mode, padding_mode, align_corners)(a, grid)