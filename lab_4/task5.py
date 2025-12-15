import numpy as np
import matplotlib.pyplot as plt
from config import FunctionConfig, SystemConfig, SizeConfig, ToleranceConfig, IterationConfig, GeneralConfig
from utils import print_iteration_header, print_iteration


def F(X):
    return FunctionConfig.nonlinear_system(X)


def J(X):
    return FunctionConfig.nonlinear_system_jacobian(X)


def plot_system():
    x_vals = np.linspace(*SizeConfig.X_RANGE_SYSTEM, SizeConfig.PLOT_POINTS)
    y_vals1 = np.cos(x_vals + 0.5) - 2
    y_vals2 = np.arcsin(2 * x_vals + 1)

    plt.figure(figsize=GeneralConfig.PLOT_FIGSIZE)
    plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
    plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Графическое решение системы')
    plt.legend()
    plt.axis([*SizeConfig.X_RANGE_SYSTEM, *SizeConfig.Y_RANGE_SYSTEM])
    plt.show()

    return x_vals, y_vals1, y_vals2


def newton_system(F, J, X0):
    print_iteration_header("Метод Ньютона для системы")

    X = X0.copy()

    for i in range(IterationConfig.NEWTON):
        FX = F(X)
        norm_F = np.linalg.norm(FX)

        # Используем общую функцию вывода итерации
        x_vector = np.array([X[0], X[1]])
        print_iteration(i + 1, x_vector, norm_F)

        if norm_F < ToleranceConfig.NEWTON:
            return X, i + 1

        JX = J(X)
        dX = np.linalg.solve(JX, -FX)
        X += dX

    return X, IterationConfig.NEWTON


def run_task5():
    print("Решение системы нелинейных уравнений")

    x_vals, y_vals1, y_vals2 = plot_system()

    X0 = SystemConfig.get_task5_initial_guess()
    print(f"Начальное приближение: x0 = {X0[0]}, y0 = {X0[1]}")

    X_sol, iter_newton = newton_system(F, J, X0)

    print(f"\nРешение: x = {X_sol[0]:.6f}, y = {X_sol[1]:.6f}")
    print(f"Итераций: {iter_newton}")

    F_sol = F(X_sol)
    print(f"\nПроверка:")
    print(f"f1(x,y) = {F_sol[0]:.2e}")
    print(f"f2(x,y) = {F_sol[1]:.2e}")

    plt.figure(figsize=GeneralConfig.PLOT_FIGSIZE)
    plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
    plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
    plt.plot(X_sol[0], X_sol[1], 'go', markersize=10, label=f'Решение ({X_sol[0]:.3f}, {X_sol[1]:.3f})')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Решение системы нелинейных уравнений')
    plt.legend()
    plt.axis([*SizeConfig.X_RANGE_SYSTEM, *SizeConfig.Y_RANGE_SYSTEM])
    plt.show()

    return X_sol