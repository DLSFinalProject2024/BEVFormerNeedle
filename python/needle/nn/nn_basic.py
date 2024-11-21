"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def setToTargetDim(var, target):
    original_ndim = len(var.shape)
    new_ndim      = len(target.shape)

    ret_shape = list(var.shape)
    extra_dims = new_ndim - original_ndim
    if extra_dims > 0:
        for i in range(extra_dims):
            ret_shape.append(1)
    return tuple(ret_shape)


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, dtype=dtype, device=device))
        #self.weight = Parameter(self.weight, dtype=dtype)
        if bias:
            self.bias = Parameter(init.kaiming_uniform(fan_in=out_features, fan_out=1, dtype=dtype, device=device).reshape((1, out_features)))
            '''
            self.bias = init.kaiming_uniform(fan_in=out_features, fan_out=1, dtype=dtype)
            self.bias = ops.reshape(self.bias, (1, -1))
            self.bias = Parameter(self.bias, dtype=dtype)
            '''
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        #X_reshape = ops.reshape(X, (X.shape[0], -1))
        #out = ops.matmul(X_reshape, self.weight)
        if self.bias is not None:
            bias_broad = ops.broadcast_to(self.bias, shape=out.shape)
            out_res = ops.add(out, bias_broad)
            return out_res
        else:
            return out
        ### END YOUR SOLUTION

class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        reduced_size = 1
        for shape_val in X.shape:
            reduced_size = reduced_size*shape_val
        reduced_size = reduced_size//X.shape[0]
        out = ops.reshape(X, (X.shape[0], reduced_size))
        return out
        ### END YOUR SOLUTION

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)

        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        k = logits.shape[1]
        n = logits.shape[0]
        logsumexp = ops.logsumexp(logits, axes=1)
        y_one_hot = init.one_hot(k, y, device=logits.device, dtype=logits.dtype, requires_grad=False)
        neg_part = ops.summation(ops.multiply(logits, y_one_hot), axes=(1,))
        neg_part_minus = ops.mul_scalar(neg_part, -1)
        pos_neg_part = ops.add(logsumexp, neg_part_minus)
        pos_neg_sum = ops.summation(pos_neg_part, axes=(0,))
        pos_neg_mean = ops.divide_scalar(pos_neg_sum, n)
        return pos_neg_mean
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(*(1, dim), device=device, dtype=dtype)
        self.weight = Parameter(self.weight, dtype=dtype, device=device)
        self.bias   = init.zeros(*(1, dim), device=device, dtype=dtype)
        self.bias   = Parameter(self.bias, dtype=dtype, device=device)
        self.running_mean   = init.zeros(*(dim,), device=device, dtype=dtype).data
        self.running_var    = init.ones(*(dim,), device=device, dtype=dtype).data
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, d = x.shape
        restore_shape = ops.restoreShape(x.shape, input_axes=(0,))

        if self.training is False:
            ux            = ops.reshape(self.running_mean, restore_shape)
            ux_broad      = ops.broadcast_to(ux, x.shape)
            x_minus_ux    = x - ux_broad
            var_resh      = ops.reshape(self.running_var, restore_shape)
            var_broad     = ops.broadcast_to(var_resh, x.shape)
            var_eps       = ops.add_scalar(var_broad, self.eps)
            var_sqrt      = ops.power_scalar(var_eps, 0.5)
            x_div_var     = ops.divide(x_minus_ux, var_sqrt)
            weight_broad  = ops.broadcast_to(self.weight, x_div_var.shape)
            bias_broad    = ops.broadcast_to(self.bias, x_div_var.shape)
            wx            = ops.multiply(weight_broad, x_div_var)
            wx_b          = ops.add(wx, bias_broad)
        else:
            usum          = ops.summation(x, axes=(0,))
            ux_div        = ops.divide_scalar(usum, n)
            ux            = ops.reshape(ux_div, restore_shape)
            ux_broad      = ops.broadcast_to(ux, x.shape)
            x_minus_ux    = x - ux_broad
            x_minus_sq    = ops.multiply(x_minus_ux, x_minus_ux)
            var_sum       = ops.summation(x_minus_sq, axes=(0,))
            var_div       = ops.divide_scalar(var_sum, n)
            var_resh      = ops.reshape(var_div, restore_shape)
            var_broad     = ops.broadcast_to(var_resh, x.shape)
            var_eps       = ops.add_scalar(var_broad, self.eps)
            var_sqrt      = ops.power_scalar(var_eps, 0.5)
            x_div_var     = ops.divide(x_minus_ux, var_sqrt)
            weight_broad  = ops.broadcast_to(self.weight, x_div_var.shape)
            bias_broad    = ops.broadcast_to(self.bias, x_div_var.shape)
            wx            = ops.multiply(weight_broad, x_div_var)
            wx_b          = ops.add(wx, bias_broad)

            self.running_mean = self.running_mean*(1-self.momentum) + ux.data.reshape((self.dim,)) * self.momentum
            self.running_var  = self.running_var*(1-self.momentum) + var_resh.data.reshape((self.dim,))*self.momentum
            self.running_mean = self.running_mean.reshape((self.dim,))
            self.running_var  = self.running_var.reshape((self.dim,))
        return wx_b
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(*(1, dim))
        self.weight = Parameter(self.weight, dtype=dtype, device=device)
        self.bias   = init.zeros(*(1, dim))
        self.bias   = Parameter(self.bias, dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #n, d = x.shape
        restore_shape = ops.restoreShape(x.shape, input_axes=(1,))
        usum          = ops.summation(x, axes=(1,))
        ux_div        = ops.divide_scalar(usum, self.dim)
        ux            = ops.reshape(ux_div, restore_shape)
        ux_broad      = ops.broadcast_to(ux, x.shape)
        x_minus_ux    = x - ux_broad
        x_minus_sq   = ops.multiply(x_minus_ux, x_minus_ux)
        var_sum      = ops.summation(x_minus_sq, axes=(1,))
        var_div      = ops.divide_scalar(var_sum, self.dim)
        var_resh     = ops.reshape(var_div, restore_shape)
        var_broad    = ops.broadcast_to(var_resh, x.shape)
        var_eps      = ops.add_scalar(var_broad, self.eps)
        var_sqrt     = ops.power_scalar(var_eps, 0.5)
        x_div_var    = ops.divide(x_minus_ux, var_sqrt)
        weight_broad = ops.broadcast_to(self.weight, x_div_var.shape)
        bias_broad   = ops.broadcast_to(self.bias, x_div_var.shape)
        wx           = ops.multiply(weight_broad, x_div_var)
        wx_b         = ops.add(wx, bias_broad)
        return wx_b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training is False:
            return x
        else:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            return ops.multiply(ops.mul_scalar(x, 1/(1-self.p)), mask)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.add(self.fn(x), x)
        ### END YOUR SOLUTION

