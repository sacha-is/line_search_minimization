import numpy as np
from scipy.optimize import minimize

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, 
                t0=1.0, mu=10.0, tol=1e-6, max_outer=20, max_inner=100):
    """
    Log-barrier interior point method for constrained minimization.
    """
    x = np.array(x0, dtype=float)
    t = t0
    m = len(ineq_constraints)
    path = [x.copy()]
    obj_values = []

    def barrier_obj(x, t):
        penalty = 0
        for g in ineq_constraints:
            val = g(x)
            if val <= 0 or np.isnan(val):
                return 1e20 # if not feasible
            penalty -= np.log(val)
        obj = t * func(x) + penalty
        if np.isnan(obj) or np.isinf(obj):
            return 1e20
        return obj

    for outer_iter in range(max_outer):
        constraints = []
        if eq_constraints_mat is not None and eq_constraints_rhs is not None:
            def eq_con(x):
                return eq_constraints_mat @ x - eq_constraints_rhs
            constraints.append({'type': 'eq', 'fun': eq_con})

        res = minimize(
            lambda x: barrier_obj(x, t),
            x,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': max_inner, 'ftol': tol, 'disp': False}
        )
        x = res.x
        path.append(x.copy())
        obj_values.append(func(x))

        # Stopping criterion: duality gap
        gap = m / t
        if gap < tol:
            break
        t *= mu

    return x, path, obj_values
