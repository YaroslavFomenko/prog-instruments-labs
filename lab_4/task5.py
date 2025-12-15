import numpy as np
import matplotlib.pyplot as plt
from config import (
    GeneralConfig, SystemConfig, FunctionConfig,
    SizeConfig, ToleranceConfig, IterationConfig,
    FormatConfig
)


def F(X):
    """Система нелинейных уравнений"""
    return FunctionConfig.nonlinear_system(X)


def J(X):
    """Якобиан системы"""
    return FunctionConfig.nonlinear_system_jacobian(X)


def plot_system():
    """Построение графиков системы уравнений"""
    x_min, x_max = SizeConfig.X_RANGE_SYSTEM
    x_vals = np.linspace(x_min, x_max, SizeConfig.PLOT_POINTS)

    # Первое уравнение: cos(x+0.5) - y = 2 → y = cos(x+0.5) - 2
    y_vals1 = np.cos(x_vals + 0.5) - 2

    # Второе уравнение: sin(y) - 2x = 1 → y = arcsin(2x + 1)
    y_vals2 = np.arcsin(2 * x_vals + 1)

    y_min, y_max = SizeConfig.Y_RANGE_SYSTEM

    plt.figure(figsize=GeneralConfig.PLOT_FIGSIZE)
    plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
    plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Графическое решение системы нелинейных уравнений')
    plt.legend()
    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()

    return x_vals, y_vals1, y_vals2


def print_newton_header():
    """Выводит заголовок для метода Ньютона"""
    print("Метод Ньютона для системы:")
    print("Итерация\tx\t\ty\t\tНорма F(X)")
    print("-" * FormatConfig.SEPARATOR_WIDTH)


def print_newton_iteration(iteration, X, norm_F):
    """Выводит результат итерации метода Ньютона"""
    print(
        f"{iteration}\t\t{X[0]:.{GeneralConfig.VECTOR_PRECISION}f}\t{X[1]:.{GeneralConfig.VECTOR_PRECISION}f}\t{norm_F:.{GeneralConfig.VECTOR_PRECISION}f}")


def newton_system(F, J, X0):
    """Метод Ньютона для системы нелинейных уравнений"""
    print_newton_header()

    X = X0.copy()

    for i in range(IterationConfig.NEWTON):
        FX = F(X)
        norm_F = np.linalg.norm(FX)
        print_newton_iteration(i + 1, X, norm_F)

        if norm_F < ToleranceConfig.NEWTON:
            return X, i + 1

        JX = J(X)
        dX = np.linalg.solve(JX, -FX)
        X += dX

    return X, IterationConfig.NEWTON


def print_initial_guess(X0):
    """Выводит начальное приближение"""
    print(
        f"Начальное приближение: x0 = {X0[0]:.{GeneralConfig.VECTOR_PRECISION}f}, y0 = {X0[1]:.{GeneralConfig.VECTOR_PRECISION}f}")


def print_solution(X_sol, iter_newton):
    """Выводит решение системы"""
    print(
        f"\nРешение: x = {X_sol[0]:.{GeneralConfig.VECTOR_PRECISION}f}, y = {X_sol[1]:.{GeneralConfig.VECTOR_PRECISION}f}")
    print(f"Итераций: {iter_newton}")


def check_solution(X_sol):
    """Проверяет решение подстановкой"""
    F_sol = F(X_sol)
    f1, f2 = F_sol

    print(f"\nПроверка:")
    print(f"f1(x,y) = cos({X_sol[0]:.3f}+0.5) - {X_sol[1]:.3f} - 2 = {f1:.2e}")
    print(f"f2(x,y) = sin({X_sol[1]:.3f}) - 2*{X_sol[0]:.3f} - 1 = {f2:.2e}")

    norm_F = np.linalg.norm(F_sol)
    print(f"Норма вектора невязки: {norm_F:.{GeneralConfig.VECTOR_PRECISION}e}")


def plot_final_solution(x_vals, y_vals1, y_vals2, X_sol):
    """Строит финальный график с решением"""
    x_min, x_max = SizeConfig.X_RANGE_SYSTEM
    y_min, y_max = SizeConfig.Y_RANGE_SYSTEM

    plt.figure(figsize=GeneralConfig.PLOT_FIGSIZE)
    plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
    plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
    plt.plot(X_sol[0], X_sol[1], 'go', markersize=10,
             label=f'Решение ({X_sol[0]:.3f}, {X_sol[1]:.3f})')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Решение системы нелинейных уравнений')
    plt.legend()
    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()


def run_task5():
    """Выполнение задания 5"""
    print("Решение системы нелинейных уравнений")

    x_vals, y_vals1, y_vals2 = plot_system()

    # Получаем начальное приближение из конфигурации
    X0 = SystemConfig.get_task5_initial_guess()
    print_initial_guess(X0)

    X_sol, iter_newton = newton_system(F, J, X0)

    print_solution(X_sol, iter_newton)
    check_solution(X_sol)

    plot_final_solution(x_vals, y_vals1, y_vals2, X_sol)

    return X_sol