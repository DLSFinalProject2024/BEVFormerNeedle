from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

def restoreShape(input_shape, input_axes):
    restore_shape = list(input_shape)
    if input_axes == None:
        restore_shape = tuple([1 for x in input_shape])
    else:
        if isinstance(input_axes, tuple):
            for i in input_axes:
                restore_shape[i] = 1
        else:
            restore_shape[input_axes] = 1
    return restore_shape

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Zmax = Z.max(axis=self.axes, keepdims=True) 

        restore_shape = restoreShape(Z.shape, self.axes)
        Zmax_reshaped = array_api.reshape(Zmax, restore_shape)
        Z_minus_Zmax = Z - Zmax_reshaped.broadcast_to(Z.shape)
        Zexp = array_api.exp(Z_minus_Zmax)
        Zsum = array_api.sum(Zexp, axis=self.axes)
        Zres = array_api.log(Zsum) + array_api.reshape(Zmax, Zsum.shape)
        return Zres
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        restore_shape = restoreShape(Z.shape, self.axes)
        l_reshape = reshape(node, restore_shape)
        out_grad_reshape = reshape(out_grad, restore_shape)
        res = exp(Z - l_reshape.broadcast_to(Z.shape))
        result = multiply(res, out_grad_reshape.broadcast_to(res.shape))
        return result
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

