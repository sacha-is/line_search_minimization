import numpy as np

def make_quadratic(Q):
    def f(x, need_hessian):
        val = x.T @ Q @ x
        grad = 2 * Q @ x
        hess = 2 * Q if need_hessian else None
        return val, grad, hess
    return f

def rosenbrock(x, need_hessian):
    x1, x2 = x[0], x[1]
    val = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    grad = np.array([
        -400 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**2)
    ])
    hess = None
    if need_hessian:
        hess = np.array([
            [1200*x1**2 - 400*x2 + 2, -400*x1],
            [-400*x1, 200]
        ])
    return val, grad, hess

def linear_function(a):
    def f(x, need_hessian):
        val = a.T @ x
        grad = a
        hess = np.zeros((len(a), len(a))) if need_hessian else None
        return val, grad, hess
    return f

def corner_triangle(x, need_hessian):
    e1 = np.exp(x[0] + 3 * x[1] - 0.1)
    e2 = np.exp(x[0] - 3 * x[1] - 0.1)
    e3 = np.exp(-x[0] - 0.1)
    val = e1 + e2 + e3
    grad = np.array([e1 + e2 - e3, 3 * e1 - 3 * e2])
    hess = None
    if need_hessian:
        hess = np.array([
            [e1 + e2 + e3, 3 * e1 - 3 * e2],
            [3 * e1 - 3 * e2, 9 * e1 + 9 * e2]
        ])
    return val, grad, hess
