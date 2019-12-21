import numpy as np

from utils import Registry

from .build import DESCENT_DIRECTION_REGISTRY


@DESCENT_DIRECTION_REGISTRY.register()
class Newton(object):
    def __init__(self, cfg):
        pass


    def __call__(self, optim, x):
        return -np.matmul(np.linalg.inv(optim.hessian(x)), optim.grad(x))