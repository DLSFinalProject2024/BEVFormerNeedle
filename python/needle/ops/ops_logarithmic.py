from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        Z -= Z.max(-1, keepdims=True)
        return Z - array_api.log(array_api.exp(Z).sum(-1, keepdims=True))

    def gradient(self, out_grad, node):
        # https://indii.org/blog/gradients-of-softmax-and-logsumexp/
        # d logsoftmax(Z))/ dZ = d (Z - logsumexp(Z)) / dZ
        # = 1 - d logsumexp(Z) / dZ = 1 - softmax(Z)
        Z = node.inputs[0]
        softmax = exp(logsoftmax(Z)) # softmax(Z)
        return out_grad - softmax * array_api.sum(out_grad.numpy(), axis=-1, keepdims=True)


def logsoftmax(a):
    return LogSoftmax()(a)


# Note: `axes` - Tuple of axes to sum and take the maximum element over. 
# This uses the same conventions as `needle.ops.Summation()`
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.max_Z = None # for reuse in gradient

    def compute(self, Z):
        self.max_Z = Z.max(axis=self.axes, keepdims=True)
        diff = Z - self.max_Z.broadcast_to(Z.shape)
        e = array_api.exp(diff)
        se = array_api.sum(e, axis=self.axes) # don't keepdims
        lse = array_api.log(se)
        return lse + self.max_Z.reshape(lse.shape)
    
    def gradient(self, out_grad, node):
        Z = node.inputs[0] # Tensor
        
        ## Approach 1: Using the chain rule
        # e = exp(Z - self.max_Z)
        # se = summation(e, axes=self.axes)
        # lse = log(se)
        # grad = Log().gradient(out_grad, lse)
        # grad = Summation(axes=self.axes).gradient(grad, se)
        # grad = Exp().gradient(grad, e)
        # return grad

        ## Approach 2: https://indii.org/blog/gradients-of-softmax-and-logsumexp/
        # Construct reduced shape with dim kept
        shape = list(Z.shape)
        axes = range(len(shape)) if self.axes is None else self.axes
        for axis in axes:
            shape[axis] = 1

        # Note: by default summation will not keepdims
        # TODO: make it simpler, find out the correct shape of each term
        diff = Z - self.max_Z.broadcast_to(Z.shape)
        softmax = exp(diff) / summation(exp(diff), axes=self.axes).reshape(shape).broadcast_to(Z.shape)
        out_grad = out_grad.reshape(shape).broadcast_to(Z.shape)

        return out_grad * softmax


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

