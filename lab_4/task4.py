import numpy as np
import matplotlib.pyplot as plt
from config import (
    GeneralConfig, FunctionConfig, SizeConfig,
    ToleranceConfig, IterationConfig, FormatConfig
)


def f(x):
    """Исходная функция: f(x) = x³ - 1.89x² - 2x + 1.76"""
    return FunctionConfig.nonlinear_function(x)


def plot_function():
    """Построение графика функции для отделения корней"""
    x_min, x_max = SizeConfig.X_RANGE_NONLINEAR
    x_plot = np.linspace(x_min, x_max, SizeConfig.PLOT_POINTS)
    y_plot = f(x_plot)

    plt.figure(figsize=GeneralConfig.PLOT_FIGSIZE)
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
    print("-" * FormatConfig.SEPARATOR_WIDTH)


def print_iteration_result(iteration, x, fx):
    """Выводит результат одной итерации"""
    print(f"{iteration}\t\t{x:.{GeneralConfig.VECTOR_PRECISION}f}\t{fx:.{GeneralConfig.VECTOR_PRECISION}f}")


def bisection(f, a, b):
    """Метод половинного деления"""
    print_iteration_header("Метод половинного деления", a, b)

    for i in range(IterationConfig.BISECTION):
        c = (a + b) / 2
        fc = f(c)
        print_iteration_result(i + 1, c, fc)

        if fc == 0 or (b - a) / 2 < ToleranceConfig.BISECTION:
            return c, i + 1

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return (a + b) / 2, IterationConfig.BISECTION


def chord(f, a, b):
    """Метод хорд"""
    print_iteration_header("Метод хорд", a, b)

    x_prev = a
    x = a - f(a) * (b - a) / (f(b) - f(a))

    for i in range(IterationConfig.CHORD):
        fx = f(x)
        print_iteration_result(i + 1, x, fx)

        if abs(fx) < ToleranceConfig.CHORD or abs(x - x_prev) < ToleranceConfig.CHORD:
            return x, i + 1

        if f(a) * fx < 0:
            b = x
        else:
            a = x

        x_prev = x
        x = a - f(a) * (b - a) / (f(b) - f(a))

    return x, IterationConfig.CHORD


def analyze_function():
    """Анализирует функцию на разных интервалах"""
    print("\nАнализ функции на интервалах:")

    test_points = [-1, 0, 1, 2]
    for x in test_points:
        fx = f(x)
        sign = "положительное" if fx > 0 else "отрицательное" if fx < 0 else "ноль"
        print(f"f({x}) = {fx:.3f} ({sign})")


def print_results(x_bisect, iter_bisect, x_chord, iter_chord):
    """Выводит результаты обоих методов"""
    print(f"\nРезультаты:")
    print(f"Метод половинного деления: x = {x_bisect:.{GeneralConfig.VECTOR_PRECISION}f} (итераций: {iter_bisect})")
    print(f"Метод хорд: x = {x_chord:.{GeneralConfig.VECTOR_PRECISION}f} (итераций: {iter_chord})")
    print(f"Значение функции в корне: f(x) = {f(x_chord):.{GeneralConfig.VECTOR_PRECISION}e}")


def run_task4():
    """Выполнение задания 4"""
    print("Решение нелинейного уравнения")

    analyze_function()
    plot_function()

    # Получаем интервал из конфигурации
    a, b = FunctionConfig.get_root_interval()

    x_bisect, iter_bisect = bisection(f, a, b)
    x_chord, iter_chord = chord(f, a, b)

    print_results(x_bisect, iter_bisect, x_chord, iter_chord)

    return x_chord