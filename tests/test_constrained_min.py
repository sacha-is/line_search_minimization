import unittest
import numpy as np
from src.constrained_min import interior_pt

class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        def func(x):
            return x[0]**2 + x[1]**2 + (x[2]+1)**2

        # Inequality constraints: x >= 0, y >= 0, z >= 0
        ineq_constraints = [
            lambda x: x[0],
            lambda x: x[1],
            lambda x: x[2]
        ]

        # Equality constraint: x + y + z = 1
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])

        x0 = np.array([0.1, 0.2, 0.7])

        x_star, _, _ = interior_pt(func, ineq_constraints, A, b, x0)

        self.assertTrue(np.all(x_star >= -1e-6))
        self.assertAlmostEqual(np.sum(x_star), 1.0, places=5)

        self.assertAlmostEqual(x_star[2], 1.0, places=2)

    def test_lp(self):
        def func(x):
            return -x[0] - x[1]

        # Inequality constraints:
        # y >= -x + 1  =>  x + y - 1 >= 0
        # y <= 1       =>  1 - y >= 0
        # x <= 2       =>  2 - x >= 0
        # y >= 0
        ineq_constraints = [
            lambda x: x[0] + x[1] - 1,
            lambda x: 1 - x[1],
            lambda x: 2 - x[0],
            lambda x: x[1]
        ]

        A = None
        b = None
        x0 = np.array([0.5, 0.75])

        x_star, _, _ = interior_pt(func, ineq_constraints, A, b, x0)

        self.assertTrue(np.all([g(x_star) >= -1e-6 for g in ineq_constraints]))

        self.assertAlmostEqual(x_star[0], 2.0, places=2)
        self.assertAlmostEqual(x_star[1], 1.0, places=2)

if __name__ == '__main__':
    unittest.main()
