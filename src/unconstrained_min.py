import numpy as np

def backtracking_line_search(f, x, direction, grad, alpha=1.0, rho=0.5, c=0.01):
    """Backtracking line search satisfying the Wolfe condition."""
    fx, _, _ = f(x, False)
    while True:
        new_x = x + alpha * direction
        new_fx, _, _ = f(new_x, False)
        if new_fx <= fx + c * alpha * grad.T @ direction:
            break
        alpha *= rho
    return alpha

def gradient_descent(f, x0, obj_tol, param_tol, max_iter):
    """Gradient descent with backtracking line search."""
    x = x0.copy()
    path = [x.copy()]
    obj_values = []
    for i in range(max_iter):
        fx, grad, _ = f(x, False)
        obj_values.append(fx)

        direction = -grad
        alpha = backtracking_line_search(f, x, direction, grad)
        new_x = x + alpha * direction

        print(f"Iter {i}: x = {x}, f(x) = {fx}")
        # check if the the old line minus new x is less than the tolerance
        if np.linalg.norm(new_x - x) < param_tol or abs(fx - f(new_x, False)[0]) < obj_tol:
            return new_x, f(new_x, False)[0], True, path, obj_values
        x = new_x
        path.append(x.copy())
    return x, f(x, False)[0], False, path, obj_values

def newton_method(f, x0, obj_tol, param_tol, max_iter):
    """Newton's method with backtracking line search."""
    x = x0.copy()
    path = [x.copy()]
    obj_values = []
    for i in range(max_iter):
        fx, grad, hess = f(x, True)
        obj_values.append(fx)

        try:
            direction = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessian not invertible")
            return x, fx, False, path, obj_values

        alpha = backtracking_line_search(f, x, direction, grad)
        new_x = x + alpha * direction

        print(f"Iter {i}: x = {x}, f(x) = {fx}")
        if np.linalg.norm(new_x - x) < param_tol or abs(fx - f(new_x, True)[0]) < obj_tol:
            return new_x, f(new_x, True)[0], True, path, obj_values
        x = new_x
        path.append(x.copy())
    return x, f(x, True)[0], False, path, obj_values
