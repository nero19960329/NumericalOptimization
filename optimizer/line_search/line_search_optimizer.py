from ..base_optimizer import BaseOptimizer
from .descent_direction import build_descent_direction
from .step_length import build_step_length


class LineSearchOptimizer(BaseOptimizer):
    def __init__(self, descent_direction_cfg, step_length_cfg, verbose=False):
        super(LineSearchOptimizer, self).__init__(verbose)

        self.descent_direction = build_descent_direction(descent_direction_cfg)
        self.step_length = build_step_length(step_length_cfg)


    def optimize(self):
        assert self.x_init is not None
        assert self.max_iter is not None

        self.x_history = [ self.x_init ]
        x = self.x_init.copy()
        fx = self.func(x)
        for k in range(self.max_iter):
            p = self.descent_direction(self, x)
            alpha = self.step_length(self, x, p)
            x += alpha * p
            new_fx = self.func(x)
            self.x_history.append(x.copy())

            if abs(new_fx - fx) < self.tol:
                break

            fx = new_fx
        return x
