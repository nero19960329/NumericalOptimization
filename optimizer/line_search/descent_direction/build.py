from utils.registry import Registry


DESCENT_DIRECTION_REGISTRY = Registry("DESCENT_DIRECTION")


from .steepest_descent import SteepestDescent
from .newton import Newton


def build_descent_direction(cfg):
    return DESCENT_DIRECTION_REGISTRY.get(cfg["name"])(cfg)