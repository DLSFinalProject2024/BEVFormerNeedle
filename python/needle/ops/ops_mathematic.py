"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

def reduceSameShape(var, dvar):
    original_ndim = len(var.shape)
    summed = None
    reduced_dim = []
    for i in range(original_ndim):
        if var.shape[i] != dvar.shape[i]:
            reduced_dim.append(i)
    summed = summation(dvar, tuple(reduced_dim))
    summed_reshape = summed.reshape(var.shape)
    return summed_reshape

def setToOriginalShape(var, dvar):
    original_ndim = len(var.shape)
    new_ndim      = len(dvar.shape)

    extra_dims = new_ndim - original_ndim
    if extra_dims > 0:
        #summed = dvar.sum(axes=tuple(range(extra_dims)))
        extra_axes = tuple(range(extra_dims))
        summed = None
        if extra_axes is None:
            summed =  dvar.sum(axis = None)
        elif isinstance(extra_axes, int) or (isinstance(extra_axes, (list, tuple)) and len(extra_axes) == 1):
            return dvar.sum(extra_axes)
        else:
            for axis in reversed(sorted(extra_axes)):
                summed = dvar.sum(axis = axis)
        dvar = reduceSameShape(var, summed)
        return dvar.reshape(var.shape)
    else:
        dvar = reduceSameShape(var, dvar)
        return dvar


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        #input('add1')
        #input('add2')
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        #input('mul_scalar1')
        ret_grad = (out_grad * self.scalar,)
        #input('mul_scalar2')
        return ret_grad


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        #return array_api.power(a, b)
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        b = node.inputs[1]
        return multiply(out_grad, multiply(b, power(a, b-Tensor(1)))), multiply(power(a, b), log(a))
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        #return array_api.power(a, self.scalar)
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad*self.scalar*power_scalar(a, self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, - a * out_grad / b ** 2        
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #input('div_scalar1')
        ret_grad = out_grad*1/self.scalar
        #input('div_scalar2')
        return ret_grad
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        a_dim_length = len(a.shape)
        axes_length = len(self.axes) if self.axes is not None else 0
        if self.axes is None:
            new_axes = tuple(x for x in range(a_dim_length-2)) + (a_dim_length-1, a_dim_length-2)
        else:
            if axes_length < a_dim_length:
                new_axes = [x for x in range(a_dim_length)]
                new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
                new_axes = tuple(new_axes)
            else:
                new_axes = self.axes[0:a_dim_length]
                new_axes = new_axes[::-1]
        return a.permute(new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #input('bc1')
        a = node.inputs[0]
        if out_grad.shape != a.shape:
            out_grad = setToOriginalShape(a, out_grad)

        #input('bc2')
        return out_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum(axis = None)
        elif isinstance(self.axes, int) or (isinstance(self.axes, (list, tuple)) and len(self.axes) == 1):
            return a.sum(self.axes)
        else:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a        
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #input('sum1')
        a = node.inputs[0]
        restore_shape = list(out_grad.shape)
        if self.axes == None:
            restore_shape = tuple(1 for x in a.shape)
        else:
            if isinstance(self.axes, tuple):
                for i in self.axes:
                    restore_shape.insert(i, 1)
            else:
                restore_shape.insert(self.axes, 1)

        out_grad = out_grad.reshape(restore_shape)
        result = broadcast_to(out_grad, a.shape)

        #input('sum2')
        return result
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #input('matmul1')
        a, b = node.inputs
        da = matmul(out_grad, transpose(b))
        db = matmul(transpose(a), out_grad)

        if da.shape != a.shape:
            da = setToOriginalShape(a, da)

        if db.shape != b.shape:
            db = setToOriginalShape(b, db)

        #input('matmul2')
        return da, db
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #input('divide1')
        grad_a = multiply(out_grad, divide(Tensor(1), node.inputs[0]))
        #input('divide2')
        return grad_a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0, device=out_grad.device)
        '''        
        a = node.inputs[0]
        grad_relu = array_api.zeros(a.shape)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a.cached_data[i][j] > 0:
                    grad_relu[i][j] = 1
        grad_relu = Tensor(grad_relu, dtype=a.dtype)
        res = multiply(out_grad, grad_relu)
        return res
        '''
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        '''
        return out_grad * (init.ones(*out_grad.shape,
                                     device=out_grad.device,
                                     requires_grad=False) - power_scalar(tanh(node.inputs[0]), 2.))        
        '''
        a = node.inputs[0]
        return out_grad * mul_scalar(power_scalar(exp(-a)+exp(a), -2), 4)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(args) == 0:
            raise ValueError("Stack needs at least one array!")

        base_shape = args[0].shape
        for arr in args:
            if arr.shape != base_shape:
                raise ValueError("All arrays need to be of the same size!")

        num_arrays = len(args)
        output_shape = list(base_shape)
        output_shape.insert(self.axis, num_arrays)

        output = array_api.empty(output_shape, device=args[0].device)
        indices = [slice(0, dim, 1) for dim in output_shape]

        # Stack along the axis
        for idx, array in enumerate(args):
            indices[self.axis] = idx
            output[tuple(indices)] = array
        return output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # Get the size along the stacking axis
        stack_size = A.shape[self.axis]
    
        # Create a new shape excluding the stacking axis
        new_shape = A.shape[:self.axis] + A.shape[self.axis + 1:]
    
        # Initialize a list to hold the split parts
        split_parts = []
        index_slices = [slice(0, dim) for dim in A.shape]
    
        # Create the slices for each split along the axis
        for index in range(stack_size):
            # Create a slice object for each index along the axis
            index_slices[self.axis] = index
        
            # Extract and reshape the part and add it to split_parts
            part = A[tuple(index_slices)].compact().reshape(new_shape)
            split_parts.append(part)
    
        return tuple(split_parts)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_offset = 0
        new_stride = [a_stride for a_stride in a.strides]
        new_shape = (a_shape for a_shape in a.shape)
        if isinstance(self.axes, tuple):
            for axes_val in self.axes:
                new_offset += (a.shape[axes_val]-1)*(a.strides[axes_val])
                new_stride[axes_val] *= -1
        else:
            new_offset += (a.shape[self.axes]-1)*(a.strides[self.axes])
            new_stride[self.axes] *= -1

        new_stride = tuple(new_stride)
        return NDArray.make(new_shape, new_stride, a.device, a._handle, new_offset).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Flip(self.axes)(out_grad)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a

        output_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            output_shape[axis] = a.shape[axis] + (a.shape[axis])*self.dilation

        output_array = array_api.full(output_shape, 0, a.dtype, a.device)
        slices = [slice(None)]*len(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, None, self.dilation+1)

        output_array[tuple(slices)] = a
        return output_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(None)]*len(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, None, self.dilation+1)

        output_arr = a[tuple(slices)]
        return output_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A_pad = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in  = A_pad.shape #N x H x W x C_in
        K, _, _, C_out = B.shape #K x K x C_in x C_out
        Ns, Hs, Ws, Cs = A_pad.strides

        inner_dim = K*K*C_in
        H_out = (H-K+1)//self.stride
        W_out = (W-K+1)//self.stride
        A_stride = A_pad.as_strided(shape   = (N , H_out         , W_out         , K , K , C_in),
                                    strides = (Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs  )).compact().reshape((N*H_out*W_out, inner_dim))
        B_reshape = B.compact().reshape((inner_dim, C_out)) 
        out = A_stride @ B_reshape
        return out.compact().reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X = node.inputs[0] #A, N x H x W x C_in
        W = node.inputs[1] #B, K x K x C_in x C_out
        K, K_, C_in, C_out = W.shape
        P = self.padding

        #------------------ Grad X---------------------#
        # Dilate
        out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride-1) #N x (H+2P-K+1) x (W+2P-K+1) x C_out 

        # Flip the kernel vertically and horizontally
        W_kernel_flip = flip(W, (0, 1)) # K x K x C_in x C_out
        W_trans       = transpose(W_kernel_flip, (2, 3)) # K x K x C_out x C_in

        # Convolution: gradient of X = out_grad @ W.T
        grad_X = conv(out_grad, W_trans, stride=1, padding=K-1-P)
        #------------------ Grad W---------------------#
        # Make summation along Batches: N
        X_trans = transpose(X, (0, 3)) # C_in x H x W x N
        out_grad_trans = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H+2P-K+1) x (W+2P-K+1) x N x C_out

        # Convolution: gradient of W = X.T @ out_grad
        grad_W_trans = conv(X_trans, out_grad_trans, stride=1, padding=self.padding) # C_in x K x K x C_out
        grad_W = transpose(transpose(grad_W_trans, (0, 1)), (1, 2)) #K x K x C_in x C_out
        return grad_X, grad_W
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


