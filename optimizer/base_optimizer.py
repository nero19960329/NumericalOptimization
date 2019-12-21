class BaseOptimizer(object):
    def __init__(self, verbose=False):
        self.func = None
        self.grad = None
        self.hessian = None
        self.x_init = None
        self.max_iter = None

        self.verbose = verbose


    def set_function(self, func):
        """
            Args:
                func: The objective function that returns a scalar y.
        """
        self.func = func


    def set_grad(self, grad):
        """
            Args:
                grad: Gradient of objective function that returns a n-dim vector.
        """
        self.grad = grad


    def set_hessian(self, hessian):
        """
            Args:
                hessian: Hessian of objective function that returns a nxn matrix.
        """
        self.hessian = hessian


    def set_x_init(self, x_init):
        self.x_init = x_init


    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    
    def set_tol(self, tol):
        self.tol = tol


    def optimize(self):
        raise NotImplementedError