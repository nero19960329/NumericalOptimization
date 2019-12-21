from utils.registry import Registry


STEP_LENGTH_REGISTRY = Registry("STEP_LENGTH")


from .backtracking import BackTracking


def build_step_length(cfg):
    return STEP_LENGTH_REGISTRY.get(cfg["name"])(cfg)