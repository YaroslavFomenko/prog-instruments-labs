import numpy as np
from task1 import householder_qr
from config import (
    GeneralConfig, SystemConfig,
    ToleranceConfig, FormatConfig
)


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
    print("Матрица A:\n", np.round(A, GeneralConfig.MATRIX_PRECISION))
    print("Вектор b:", np.round(b, GeneralConfig.VECTOR_PRECISION))


def print_solution_comparison(A, b, x_qr):
    """Сравнивает решение с numpy и вычисляет невязку"""
    # Проверка с помощью numpy
    x_np = np.linalg.solve(A, b)

    print(f"\nРешение методом QR-разложения: x = {np.round(x_qr, GeneralConfig.VECTOR_PRECISION)}")
    print(f"Решение numpy: x = {np.round(x_np, GeneralConfig.VECTOR_PRECISION)}")

    # Разность норм
    diff_norm = np.linalg.norm(x_qr - x_np)
    print(f"Разность норм: {diff_norm:.{GeneralConfig.VECTOR_PRECISION}f}")

    # Проверка невязки
    residual = A @ x_qr - b
    residual_norm = np.linalg.norm(residual)
    print(f"Норма невязки (Ax - b): {residual_norm:.{GeneralConfig.VECTOR_PRECISION}f}")
    print(f"Вектор невязки: {np.round(residual, GeneralConfig.VECTOR_PRECISION)}")


def run_task2():
    """Выполнение задания 2"""
    # Получаем систему из конфигурации
    A, b = SystemConfig.get_task2_system()

    print_system_info(A, b)

    # Решение методом QR-разложения
    x_qr, Q, R = solve_with_qr(A, b)

    print_solution_comparison(A, b, x_qr)

    return x_qr