from utils import Registry

from .build import DESCENT_DIRECTION_REGISTRY


@DESCENT_DIRECTION_REGISTRY.register()
class SteepestDescent(object):
    def __init__(self, cfg):
        pass


    def __call__(self, optim, x):
        return -optim.grad(x)