import numpy as np
import matplotlib.pyplot as plt
from config import FunctionConfig, SizeConfig, ToleranceConfig, IterationConfig, GeneralConfig
from utils import print_iteration_header, print_iteration


def f(x):
    return FunctionConfig.nonlinear_function(x)


def plot_function():
    x_plot = np.linspace(*SizeConfig.X_RANGE_NONLINEAR, SizeConfig.PLOT_POINTS)
    y_plot = f(x_plot)

    plt.figure(figsize=GeneralConfig.PLOT_FIGSIZE)
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции')
    plt.legend()
    plt.show()


def bisection(f, a, b):
    print_iteration_header("Бисекция", a, b)

    for i in range(IterationConfig.BISECTION):
        c = (a + b) / 2
        fc = f(c)
        print_iteration(i + 1, c, fc)

        if fc == 0 or (b - a) / 2 < ToleranceConfig.BISECTION:
            return c, i + 1

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return (a + b) / 2, IterationConfig.BISECTION


def chord(f, a, b):
    print_iteration_header("Метод хорд", a, b)

    x_prev = a
    x = a - f(a) * (b - a) / (f(b) - f(a))

    for i in range(IterationConfig.CHORD):
        fx = f(x)
        print_iteration(i + 1, x, fx)

        if abs(fx) < ToleranceConfig.CHORD or abs(x - x_prev) < ToleranceConfig.CHORD:
            return x, i + 1

        if f(a) * fx < 0:
            b = x
        else:
            a = x

        x_prev = x
        x = a - f(a) * (b - a) / (f(b) - f(a))

    return x, IterationConfig.CHORD


def run_task4():
    print("Решение нелинейного уравнения")

    test_points = [-1, 0, 1, 2]
    for x in test_points:
        print(f"f({x}) = {f(x):.3f}")

    plot_function()

    a, b = FunctionConfig.get_root_interval()

    x_bisect, iter_bisect = bisection(f, a, b)
    x_chord, iter_chord = chord(f, a, b)

    print(f"\nРезультаты:")
    print(f"Бисекция: x = {x_bisect:.5f} ({iter_bisect} итераций)")
    print(f"Хорд: x = {x_chord:.5f} ({iter_chord} итераций)")
    print(f"f(корень) = {f(x_chord):.2e}")

    return x_chord