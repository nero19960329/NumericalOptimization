import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from optimizer import LineSearchOptimizer



def sqr(x):
    return x * x


def rosenbrock(x):
    return 100.0 * sqr(x[1] - sqr(x[0])) + sqr(1.0 - x[0])


def rosenbrock_grad(x):
    t = x[1] - sqr(x[0])
    return np.array([
        -400.0 * x[0] * t - 2.0 * (1.0 - x[0]),
        200.0 * t
    ])


def rosenbrock_hessian(x):
    return np.array([
        [ -400.0 * (x[1] - 3.0 * sqr(x[0])) + 2.0, -400.0 * x[0] ],
        [ -400.0 * x[0], 200.0 ]
    ])


optim = LineSearchOptimizer(
    {
        #"name": "SteepestDescent"
        "name": "Newton"
    },
    {
        "name": "BackTracking",
        "alpha_init": 1.0,
        "c": 1e-4,
        "rho": 0.5
    },
    verbose=True
)
optim.set_function(rosenbrock)
optim.set_grad(rosenbrock_grad)
optim.set_hessian(rosenbrock_hessian)
optim.set_x_init(np.array([-1.2, 1.0]))
optim.set_max_iter(1000)
optim.set_tol(1e-14)
optim.optimize()

figure = plt.figure()
ax = Axes3D(figure)
X = np.arange(-1.3, 1.1, 0.1)
Y = np.arange(0.0, 2.0, 0.1)
X, Y = np.meshgrid(X, Y)
XY = np.stack([X, Y]).reshape(2, -1)
Z = rosenbrock(XY).reshape(X.shape[0], X.shape[1])
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='gray')
print(optim.x_history)
x_iters = np.transpose(np.stack(optim.x_history), (1, 0))
print(x_iters, x_iters.shape)
f_iters = rosenbrock(x_iters)
print(f_iters, f_iters.shape)
ax.plot3D(x_iters[0], x_iters[1], f_iters, 'gray')
ax.scatter3D(x_iters[0], x_iters[1], f_iters)
plt.show()