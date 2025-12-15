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


def print_iteration_header(method_name, a, b):
    """Выводит заголовок для итерационного метода"""
    print(f"\n{method_name} на интервале [{a}, {b}]:")
    print("Итерация\tx\t\tf(x)")
    print("-" * 40)


def print_iteration_result(iteration, x, fx):
    """Выводит результат одной итерации"""
    print(f"{iteration}\t\t{x:.6f}\t{fx:.6f}")


def bisection(f, a, b, tol=1e-3):
    """Метод половинного деления"""
    print_iteration_header("Метод половинного деления", a, b)

    for i in range(100):
        c = (a + b) / 2
        fc = f(c)
        print_iteration_result(i + 1, c, fc)

        if fc == 0 or (b - a) / 2 < tol:
            return c, i + 1

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return (a + b) / 2, i + 1


def chord(f, a, b, tol=1e-4):
    """Метод хорд"""
    print_iteration_header("Метод хорд", a, b)

    x_prev = a
    x = a - f(a) * (b - a) / (f(b) - f(a))

    for i in range(100):
        fx = f(x)
        print_iteration_result(i + 1, x, fx)

        if abs(fx) < tol or abs(x - x_prev) < tol:
            return x, i + 1

        if f(a) * fx < 0:
            b = x
        else:
            a = x

        x_prev = x
        x = a - f(a) * (b - a) / (f(b) - f(a))

    return x, i + 1


def analyze_function():
    """Анализирует функцию на разных интервалах"""
    print("\nАнализ функции на интервалах:")
    print(f"f(-1) = {f(-1):.3f}")
    print(f"f(0) = {f(0):.3f}")
    print(f"f(1) = {f(1):.3f}")
    print(f"f(2) = {f(2):.3f}")


def print_results(x_bisect, iter_bisect, x_chord, iter_chord):
    """Выводит результаты обоих методов"""
    print(f"\nРезультаты:")
    print(f"Метод половинного деления: x = {x_bisect:.5f} (итераций: {iter_bisect})")
    print(f"Метод хорд: x = {x_chord:.5f} (итераций: {iter_chord})")
    print(f"f(x_корень) = {f(x_chord):.2e}")


def run_task4():
    """Выполнение задания 4"""
    print("Решение нелинейного уравнения")

    analyze_function()
    plot_function()

    x_bisect, iter_bisect = bisection(f, 0, 2)
    x_chord, iter_chord = chord(f, 0, 2)

    print_results(x_bisect, iter_bisect, x_chord, iter_chord)

    return x_chord