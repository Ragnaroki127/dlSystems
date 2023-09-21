"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce


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
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = X.shape[0]
        if self.bias:
            return X @ self.weight + self.bias.broadcast_to((N, self.out_features))
        else:
            return X @ self.weight
        ### END YOUR SOLUTION

class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        N = X.shape[0]
        out_dims = reduce(lambda x, y: x * y, X.shape[1:])
        return X.reshape((N, int(out_dims)))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.ReLU()(x)
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
        ## BEGIN YOUR SOLUTION
        N, dim = logits.shape
        y_one_hot = init.one_hot(dim, y)
        z = ops.logsumexp(logits, axes=(1,)) / N
        z_y = logits * y_one_hot / N
        loss = ops.summation(z) - ops.summation(z_y)
        return loss
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        self.running_mean = init.zeros(dim, requires_grad=False)
        self.running_var = init.ones(dim, requires_grad=False)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, dim = x.shape
        if self.training:
            mean = ops.summation(x, axes=(0,)) / N
            var = ops.summation((x - mean.broadcast_to((N, dim))) * (x - mean.broadcast_to((N, dim))), axes=(0,)) / N
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            mean = mean.broadcast_to((N, dim))
            var = var.broadcast_to((N, dim))
            return self.weight.broadcast_to((N, dim)) * ((x - mean) / (var + self.eps)**0.5) + self.bias.broadcast_to((N, dim))
        else:
            norm = (x - self.running_mean.broadcast_to((N, dim))) / (self.running_var.broadcast_to((N, dim)) + self.eps)**0.5
            return self.weight.broadcast_to((N, dim)) * norm + self.bias.broadcast_to((N, dim))
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, dim = x.shape
        mean = (ops.summation(x, axes=(1,)) / dim).reshape((N, 1)).broadcast_to((N, dim))
        var = (ops.summation((x - mean) * (x - mean), axes=(1,)) / dim).reshape((N, 1)).broadcast_to((N, dim))

        return self.weight.broadcast_to((N, dim)) * ((x - mean) / (var + self.eps)**0.5) + self.bias.broadcast_to((N, dim))
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            out = x / (1 - self.p)
            mask = init.randb(*out.shape, p=self.p)
            out = out * mask
            return out
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



