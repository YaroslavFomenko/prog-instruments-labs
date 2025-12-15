import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """Исходная функция: f(x) = x³ - 1.89x² - 2x + 1.76"""
    return x ** 3 - 1.89 * x ** 2 - 2 * x + 1.76


def plot_function():
    """Построение графика функции для отделения корней"""
    x_plot = np.linspace(-1, 3, 400)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x³ - 1.89x² - 2x + 1.76')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции для отделения корней')
    plt.legend()
    plt.show()

    return x_plot, y_plot


def bisection(f, a, b, tol=1e-3):
    """Метод половинного деления"""
    print(f"\nМетод половинного деления на интервале [{a}, {b}]:")
    print("Итерация\tx\t\tf(x)")
    print("-" * 40)

    for i in range(100):
        c = (a + b) / 2
        print(f"{i + 1}\t\t{c:.6f}\t{f(c):.6f}")

        if f(c) == 0 or (b - a) / 2 < tol:
            return c, i + 1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, i + 1


def chord(f, a, b, tol=1e-4):
    """Метод хорд"""
    print(f"\nМетод хорд на интервале [{a}, {b}]:")
    print("Итерация\tx\t\tf(x)")
    print("-" * 40)

    x_prev = a
    x = a - f(a) * (b - a) / (f(b) - f(a))

    for i in range(100):
        print(f"{i + 1}\t\t{x:.6f}\t{f(x):.6f}")

        if abs(f(x)) < tol or abs(x - x_prev) < tol:
            return x, i + 1

        if f(a) * f(x) < 0:
            b = x
        else:
            a = x

        x_prev = x
        x = a - f(a) * (b - a) / (f(b) - f(a))

    return x, i + 1


def run_task4():
    """Выполнение задания 4"""
    print("Решение нелинейного уравнения")

    # Построение графика
    print("\nАнализ функции на интервалах:")
    print(f"f(-1) = {f(-1):.3f}")
    print(f"f(0) = {f(0):.3f}")
    print(f"f(1) = {f(1):.3f}")
    print(f"f(2) = {f(2):.3f}")

    # Построение графика
    plot_function()

    # Применение методов
    x_bisect, iter_bisect = bisection(f, 0, 2)
    x_chord, iter_chord = chord(f, 0, 2)

    print(f"\nРезультаты:")
    print(f"Метод половинного деления: x = {x_bisect:.5f} (итераций: {iter_bisect})")
    print(f"Метод хорд: x = {x_chord:.5f} (итераций: {iter_chord})")
    print(f"f(x_корень) = {f(x_chord):.2e}")

    return x_chord