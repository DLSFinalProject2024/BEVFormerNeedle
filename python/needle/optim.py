"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for id, w in enumerate(self.params):
            gt = ndl.Tensor(w.grad.detach(), dtype=w.dtype) + self.weight_decay*w.detach()
            if id not in self.u:
                self.u[id] = ndl.init.zeros(*w.shape, device=w.device, dtype=w.dtype)

            self.u[id] = self.momentum*self.u[id] + (1-self.momentum)*gt.detach()
            gt = self.u[id]
            w.data = w.data - self.lr*gt.detach()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        # Calculate the total norm of gradients (L2 norm)
        total_norm = 0.0
        for param in self.params:
            if param.grad is not None:
                grad = param.grad.detach().numpy().reshape(-1)
                grad_norm = np.sum([g ** 2 for g in grad]) ** 0.5
                total_norm += grad_norm

        # Calculate the clipping coefficient
        clip_coef = max_norm / (total_norm + 1e-6)  # Adding a small epsilon to avoid division by zero                
        # Clip the gradients if the total norm exceeds max_norm
        if clip_coef < 1:
            for param in self.params:
                if param.grad is not None:
                    param.grad = param.grad.detach()*clip_coef
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t+=1
        for id, w in enumerate(self.params):
            gt = ndl.Tensor(w.grad.data, dtype=w.data.dtype) + self.weight_decay*w.data
            if id not in self.m:
                self.m[id] = ndl.init.zeros(*w.shape, device=w.device)
            if id not in self.v:
                self.v[id] = ndl.init.zeros(*w.shape, device=w.device)

            self.m[id] = self.beta1*self.m[id] + (1-self.beta1)*gt
            self.v[id] = self.beta2*self.v[id] + (1-self.beta2)*(gt**2)
            mtbar = self.m[id]/(1-self.beta1**self.t)
            vtbar = self.v[id]/(1-self.beta2**self.t)
            w.data = w.data - self.lr*mtbar/(vtbar**0.5 + self.eps)
        ### END YOUR SOLUTION
