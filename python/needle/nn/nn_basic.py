"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


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

        self.weight = Parameter(
            init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype).transpose()
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        y = X @ self.weight # (batch_size (B), n_out_feats)
        if self.bias is not None:
            y += ops.broadcast_to(self.bias, y.shape)

        return y


class Flatten(Module):
    def forward(self, X):
        # (B, X_0, X_1, ...) -> (B, X_0 * X_1 * ...)
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # Assume logits.shape = (batch_size (B), num_classes (C))
        batch_size, num_classes = logits.shape
        Z_logsumexp = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(num_classes, y, device=logits.device) # shape: (B, C)
        Zy = ops.summation(logits * y_one_hot, axes=(1,)) # shape: (B,)
        return ops.summation(Z_logsumexp - Zy, axes=(0,)) / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # running_mean and running_var are not parameters, we are not computing their gradients
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_out_feats = x.shape
        if self.training:
            E_x = x.sum(axes=(0,)) / batch_size # (n_out_feats,)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E_x.numpy()
            E_x = E_x.reshape((1, n_out_feats)) # (1, n_out_feats)
            E_x = E_x.broadcast_to(x.shape) # (batch_size, n_out_feats)

            diff = x - E_x
            Var_x = (diff ** 2).sum(axes=(0,)) / batch_size # (n_out_feats,)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var_x.numpy()
            Var_x = Var_x.reshape((1, n_out_feats)) # (1, n_out_feats)
            Var_x = Var_x.broadcast_to(x.shape) # (batch_size, n_out_feats) 

            x_norm = (x - E_x) / (Var_x + self.eps) ** 0.5 # (batch_size, n_out_feats)
        else:
            mean = self.running_mean.reshape((1, n_out_feats)).broadcast_to(x.shape) # (batch_size, n_out_feats)
            var = self.running_var.broadcast_to(x.shape) # (batch_size, n_out_feats)
            x_norm = (x - mean) / (var + self.eps) ** 0.5 # (batch_size, n_out_feats)

        w = self.weight.reshape((1, n_out_feats)) # (1, n_out_feats)
        w = w.broadcast_to(x.shape) # (batch_size, n_out_feats)
        b = self.bias.reshape((1, n_out_feats)) # (1, n_out_feats)
        b = b.broadcast_to(x.shape) # (batch_size, n_out_feats)

        return w * x_norm + b


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_out_feats = x.shape
        E_x = x.sum(axes=(1,)) / n_out_feats # (batch_size,)
        E_x = E_x.reshape((batch_size, 1)) # (batch_size, 1)
        E_x = E_x.broadcast_to(x.shape) # (batch_size, n_out_feats)

        diff = x - E_x
        # Var = sum((x-mean)^2) / N
        Var_x = (diff ** 2).sum(axes=(1,)) / n_out_feats # (batch_size,)
        Var_x = Var_x.reshape((batch_size, 1)) # (batch_size, 1)
        Var_x = Var_x.broadcast_to(x.shape) # (batch_size, n_out_feats)

        x_norm = (x - E_x) / (Var_x + self.eps) ** 0.5 # (batch_size, n_out_feats)

        w = self.weight.reshape((1, n_out_feats)) # (1, n_out_feats)
        w = w.broadcast_to(x.shape) # (batch_size, n_out_feats)
        b = self.bias.reshape((1, n_out_feats)) # (1, n_out_feats)
        b = b.broadcast_to(x.shape) # (batch_size, n_out_feats)
        
        return w * x_norm + b


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        probs = init.randb(*x.shape, p=(1 - self.p)) # bit mask
        return (probs * x) / (1.0 - self.p)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
