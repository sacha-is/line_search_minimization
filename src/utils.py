import numpy as np
import matplotlib.pyplot as plt
from tests.examples import make_quadratic, rosenbrock, linear_function, corner_triangle

def plot_contour(f, xlim, ylim, path_dict=None, title=""):
    """Plot contour lines of function f and optional optimization paths."""
    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z, _, _ = f(np.array([X[i, j], Y[i, j]]), False)
            Z[i, j] = z

    plt.figure()
    cp = plt.contour(X, Y, Z, levels=30)
    plt.clabel(cp)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")

    if path_dict:
        for label, path in path_dict.items():
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], label=label, marker='o')
        plt.legend()

    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_function_values(methods_values, title="Objective Value vs Iteration"):
    """Plot function values per iteration for each method."""
    plt.figure()
    for label, values in methods_values.items():
        plt.plot(values, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def choose_algorithm():
    print("Choose an optimization algorithm:")
    print("1. Gradient Descent")
    print("2. Newton's Method")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("Invalid input. Please enter 1 or 2.")

def choose_function():
    print("Choose a function to minimize:")
    print("1. Quadratic (circle)")
    print("2. Quadratic (ellipse)")
    print("3. Rosenbrock")
    print("4. Linear")
    print("5. Corner Triangle")
    while True:
        choice = input("Enter 1-5: ").strip()
        if choice == "1":
            Q = np.eye(2)
            return make_quadratic(Q), np.array([1.0, 1.0]), "Quadratic (circle)"
        elif choice == "2":
            Q = np.array([[1, 0], [0, 100]])
            return make_quadratic(Q), np.array([1.0, 1.0]), "Quadratic (ellipse)"
        elif choice == "3":
            return rosenbrock, np.array([-1.0, 2.0]), "Rosenbrock"
        elif choice == "4":
            a = np.array([2.0, -1.0])
            return linear_function(a), np.array([1.0, 1.0]), "Linear"
        elif choice == "5":
            return corner_triangle, np.array([1.0, 1.0]), "Corner Triangle"
        print("Invalid input. Please enter a number from 1 to 5.")
