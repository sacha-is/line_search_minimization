import numpy as np
import matplotlib.pyplot as plt
from src.constrained_min import interior_pt

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


def qp_example():
    def func(x):
        return x[0]**2 + x[1]**2 + (x[2]+1)**2

    ineq_constraints = [
        lambda x: x[0],
        lambda x: x[1],
        lambda x: x[2]
    ]
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([1.0])
    x0 = np.array([0.1, 0.2, 0.7])
    
    
    x_star, path, obj_values = interior_pt(func, ineq_constraints, A, b, x0)
    path = np.array(path)

    # Feasible region (triangle) and central path in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot([1,0,0,1], [0,1,0,0], [0,0,1,0], 'k-', alpha=0.3)
    ax.plot(path[:,0], path[:,1], path[:,2], 'o-', label='Central Path')
    ax.scatter(x_star[0], x_star[1], x_star[2], c='r', label='Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('QP: Feasible region and central path')
    ax.legend()
    plt.show()

    # Objective value vs. iteration
    plt.figure()
    plt.plot(obj_values, 'o-')
    plt.xlabel('Outer iteration')
    plt.ylabel('Objective value')
    plt.title('QP: Objective value vs. iteration')
    plt.show()

    print("QP final solution:", x_star)
    print("QP objective value:", func(x_star))
    print("QP constraint (sum):", np.sum(x_star), "ineqs:", [g(x_star) for g in ineq_constraints])

def lp_example():
    def func(x):
        return -x[0] - x[1]

    ineq_constraints = [
        lambda x: x[0] + x[1] - 1,
        lambda x: 1 - x[1],
        lambda x: 2 - x[0],
        lambda x: x[1]
    ]
    A = None
    b = None
    x0 = np.array([0.5, 0.75])

    x_star, path, obj_values = interior_pt(func, ineq_constraints, A, b, x0)
    path = np.array(path)

    # Feasible region and central path in 2D
    _, ax = plt.subplots()
    # Polygon vertices: intersection of constraints
    verts = np.array([
        [2, 0],
        [2, 1],
        [0, 1],
        [1, 0]
    ])
    poly = plt.Polygon(verts, fill=None, edgecolor='k', alpha=0.3)
    ax.add_patch(poly)
    # Path
    ax.plot(path[:,0], path[:,1], 'o-', label='Central Path')
    ax.scatter(x_star[0], x_star[1], c='r', label='Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('LP: Feasible region and central path')
    ax.legend()
    plt.show()

    # Objective value vs. iteration
    plt.figure()
    plt.plot(obj_values, 'o-')
    plt.xlabel('Outer iteration')
    plt.ylabel('Objective value')
    plt.title('LP: Objective value vs. iteration')
    plt.show()

    print("LP final solution:", x_star)
    print("LP objective value:", func(x_star))
    print("LP constraints:", [g(x_star) for g in ineq_constraints])

if __name__ == "__main__":
    qp_example()
    lp_example()