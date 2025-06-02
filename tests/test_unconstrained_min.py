import unittest
import numpy as np
from src.unconstrained_min import gradient_descent, newton_method
from src.utils import plot_contour, plot_function_values
from tests.examples import make_quadratic, rosenbrock, linear_function, corner_triangle

class TestMinimization(unittest.TestCase):
    def run_test(self, f, x0, name, max_iter=100):
        obj_tol = 1e-12
        param_tol = 1e-8

        x_gd, fx_gd, success_gd, path_gd, vals_gd = gradient_descent(f, x0, obj_tol, param_tol, max_iter)
        x_nt, fx_nt, success_nt, path_nt, vals_nt = newton_method(f, x0, obj_tol, param_tol, max_iter)

        print(f"{name} GD Final: x={x_gd}, f={fx_gd}, success={success_gd}")
        print(f"{name} NT Final: x={x_nt}, f={fx_nt}, success={success_nt}")

        plot_contour(f, (-3, 3), (-3, 3), {
            "Gradient Descent": path_gd,
            "Newton": path_nt
        }, title=f"{name} Contours + Paths")

        plot_function_values({
            "Gradient Descent": vals_gd,
            "Newton": vals_nt
        }, title=f"{name} Objective Value vs Iteration")

    def test_circle(self):
        Q = np.eye(2)
        f = make_quadratic(Q)
        self.run_test(f, np.array([1.0, 1.0]), "Circle")

    def test_ellipse(self):
        Q = np.array([[1, 0], [0, 100]])
        f = make_quadratic(Q)
        self.run_test(f, np.array([1.0, 1.0]), "Ellipse")

    def test_rotated_ellipse(self):
        theta = np.pi / 6
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        Q = R.T @ np.diag([100, 1]) @ R
        f = make_quadratic(Q)
        self.run_test(f, np.array([1.0, 1.0]), "Rotated Ellipse")

    def test_rosenbrock(self):
        self.run_test(rosenbrock, np.array([-1.0, 2.0]), "Rosenbrock", max_iter=10000)

    def test_linear(self):
        a = np.array([2.0, -1.0])
        f = linear_function(a)
        self.run_test(f, np.array([1.0, 1.0]), "Linear")

    def test_corner_triangle(self):
        self.run_test(corner_triangle, np.array([1.0, 1.0]), "Corner Triangle")

if __name__ == "__main__":
    unittest.main()
