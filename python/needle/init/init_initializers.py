import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain*(math.sqrt(6/(fan_in+fan_out)))
    if shape is None:
        shape_ = (fan_in, fan_out)
        return rand(*shape_, low=-a, high=a, **kwargs)
    else:
        return rand(*shape, low=-a, high=a, **kwargs)

    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain*(math.sqrt(2/(fan_in+fan_out)))
    if shape is None:
        shape_ = (fan_in, fan_out)
        return randn(*shape_, mean=0, std=std, **kwargs)
    else:
        return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain*(math.sqrt(3/fan_in))
    if shape is None:
        if fan_out==None:
            shape_ = (fan_in,)
        else:
            shape_ = (fan_in, fan_out)
        return rand(*shape_, low=-bound, high=bound, **kwargs)
    else:
        return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain/math.sqrt(fan_in)
    if shape is None:
        shape_ = (fan_in, fan_out)
        return randn(*shape_, mean=0, std=std, **kwargs)
    else:
        return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION