import numpy as np
from src.unconstrained_min import gradient_descent, newton_method
from src.utils import choose_algorithm, choose_function, plot_contour, plot_function_values

def main():
    alg_choice = choose_algorithm()
    f, x0, fname = choose_function()
    obj_tol = 1e-12
    param_tol = 1e-8
    max_iter = 1000

    if alg_choice == 1:
        print(f"\nRunning Gradient Descent on {fname}...")
        x, fx, success, path, vals = gradient_descent(f, x0, obj_tol, param_tol, max_iter)
        plot_contour(f, (-3, 3), (-3, 3), {"Gradient Descent": path}, title=f"{fname} Contour + Path")
        plot_function_values({"Gradient Descent": vals}, title=f"{fname} Objective Value vs Iteration")
    else:
        print(f"\nRunning Newton's Method on {fname}...")
        x, fx, success, path, vals = newton_method(f, x0, obj_tol, param_tol, max_iter)
        plot_contour(f, (-3, 3), (-3, 3), {"Newton": path}, title=f"{fname} Contour + Path")
        plot_function_values({"Newton": vals}, title=f"{fname} Objective Value vs Iteration")

    print(f"\nResult: x = {x}, f(x) = {fx}, success = {success}")

if __name__ == "__main__":
    main()
