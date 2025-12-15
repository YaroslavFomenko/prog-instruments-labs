import numpy as np
from task1 import householder_qr


def back_substitution(R, b):
    """Обратная подстановка для верхней треугольной матрицы"""
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x


def solve_with_qr(A, b):
    """Решение системы линейных уравнений методом QR-разложения"""
    # QR-разложение
    Q, R = householder_qr(A)

    # Преобразование правой части: Q^T b
    b_new = Q.T @ b

    # Решение обратной подстановкой
    x = back_substitution(R, b_new)

    return x, Q, R


def print_system_info(A, b):
    """Выводит информацию о системе"""
    print("Матрица A:\n", A)
    print("Вектор b:", b)


def print_solution_comparison(A, b, x_qr):
    """Сравнивает решение с numpy и вычисляет невязку"""
    # Проверка с помощью numpy
    x_np = np.linalg.solve(A, b)
    print("\nРешение методом QR-разложения: x =", np.round(x_qr, 4))
    print("Решение numpy: x =", np.round(x_np, 4))
    print("Разность норм:", np.round(np.linalg.norm(x_qr - x_np), 6))

    # Проверка невязки
    residual = A @ x_qr - b
    print("Невязка (Ax - b):", np.round(residual, 6))


def run_task2():
    """Выполнение задания 2"""
    # Система для варианта 4
    A = np.array([
        [17.1, -8.3, 14.4, 7.2],
        [6.4, 8.5, -4.3, 8.8],
        [8.3, -6.6, 5.8, 12.2],
        [3.8, 14.2, 6.3, -15.5]
    ], dtype=float)

    b = np.array([13.5, 7.7, -4.7, 2.8], dtype=float)

    print_system_info(A, b)

    # Решение методом QR-разложения
    x_qr, Q, R = solve_with_qr(A, b)

    print_solution_comparison(A, b, x_qr)

    return x_qr