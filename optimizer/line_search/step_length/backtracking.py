import numpy as np

from utils.registry import Registry

from .build import STEP_LENGTH_REGISTRY


@STEP_LENGTH_REGISTRY.register()
class BackTracking(object):
    def __init__(self, cfg):
        self.alpha_init = cfg["alpha_init"]
        self.c = cfg["c"]
        self.rho = cfg["rho"]


    def __call__(self, optim, x, p):
        alpha = self.alpha_init
        while optim.func(x + alpha * p) > optim.func(x) + self.c * alpha * np.dot(optim.grad(x), p):
            alpha *= self.rho
        return alpha