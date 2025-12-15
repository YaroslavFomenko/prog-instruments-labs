import numpy as np
import matplotlib.pyplot as plt


def F(X):
    """Система нелинейных уравнений"""
    x, y = X
    return np.array([
        np.cos(x + 0.5) - y - 2,
        np.sin(y) - 2 * x - 1
    ])


def J(X):
    """Якобиан системы"""
    x, y = X
    return np.array([
        [-np.sin(x + 0.5), -1],
        [-2, np.cos(y)]
    ])


def plot_system():
    """Построение графиков системы уравнений"""
    x_vals = np.linspace(-2, 2, 100)
    y_vals1 = np.cos(x_vals + 0.5) - 2
    y_vals2 = np.arcsin(2 * x_vals + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
    plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Графическое решение системы нелинейных уравнений')
    plt.legend()
    plt.axis([-2, 2, -3, 1])
    plt.show()

    return x_vals, y_vals1, y_vals2


def print_newton_header():
    """Выводит заголовок для метода Ньютона"""
    print("Метод Ньютона для системы:")
    print("Итерация\tx\t\ty\t\tНорма F(X)")
    print("-" * 60)


def print_newton_iteration(iteration, X, norm_F):
    """Выводит результат итерации метода Ньютона"""
    print(f"{iteration}\t\t{X[0]:.6f}\t{X[1]:.6f}\t{norm_F:.6f}")


def newton_system(F, J, X0, tol=1e-4):
    """Метод Ньютона для системы нелинейных уравнений"""
    print_newton_header()

    X = X0.copy()

    for i in range(100):
        FX = F(X)
        norm_F = np.linalg.norm(FX)
        print_newton_iteration(i + 1, X, norm_F)

        if norm_F < tol:
            return X, i + 1

        JX = J(X)
        dX = np.linalg.solve(JX, -FX)
        X += dX

    return X, i + 1


def print_initial_guess(X0):
    """Выводит начальное приближение"""
    print(f"Начальное приближение: x0 = {X0[0]}, y0 = {X0[1]}")


def print_solution(X_sol, iter_newton):
    """Выводит решение системы"""
    print(f"\nРешение: x = {X_sol[0]:.6f}, y = {X_sol[1]:.6f}")
    print(f"Итераций: {iter_newton}")


def check_solution(X_sol):
    """Проверяет решение подстановкой"""
    F_sol = F(X_sol)
    print(f"\nПроверка:")
    print(f"f1(x,y) = cos({X_sol[0]:.3f}+0.5) - {X_sol[1]:.3f} - 2 = {F_sol[0]:.2e}")
    print(f"f2(x,y) = sin({X_sol[1]:.3f}) - 2*{X_sol[0]:.3f} - 1 = {F_sol[1]:.2e}")


def plot_final_solution(x_vals, y_vals1, y_vals2, X_sol):
    """Строит финальный график с решением"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
    plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
    plt.plot(X_sol[0], X_sol[1], 'go', markersize=10, label=f'Решение ({X_sol[0]:.3f}, {X_sol[1]:.3f})')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Решение системы нелинейных уравнений')
    plt.legend()
    plt.axis([-2, 2, -3, 1])
    plt.show()


def run_task5():
    """Выполнение задания 5"""
    print("Решение системы нелинейных уравнений")

    x_vals, y_vals1, y_vals2 = plot_system()

    X0 = np.array([0.0, -1.0])
    print_initial_guess(X0)

    X_sol, iter_newton = newton_system(F, J, X0)

    print_solution(X_sol, iter_newton)
    check_solution(X_sol)

    plot_final_solution(x_vals, y_vals1, y_vals2, X_sol)

    return X_sol