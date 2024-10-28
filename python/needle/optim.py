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
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            # u_{t+1} = \beta * u_t + (1 - \beta) * grad
            self.u[p] = self.momentum * self.u.get(p, 0) + (1 - self.momentum) * grad
            # p_{t+1} = p_t - lr * u_{t+1}
            p.data -= self.lr * self.u[p]
        

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
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
        self.t += 1
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            # u_{t+1} = \beta_1 * u_t + (1 - \beta_1) * grad
            self.m[p] = self.beta1 * self.m.get(p, 0) + (1 - self.beta1) * grad
            # v_{t+1} = \beta_2 * v_t + (1 - \beta_2) * grad^2
            self.v[p] = self.beta2 * self.v.get(p, 0) + (1 - self.beta2) * (grad ** 2)
            # u_hat = u / (1 - \beta_1^t)
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            # v_hat = v / (1 - \beta
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            # p_{t+1} = p_t - lr * u_hat / (v_hat^0.5 + eps)
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
