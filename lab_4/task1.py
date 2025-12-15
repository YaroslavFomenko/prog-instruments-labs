import numpy as np
from config import (
    GeneralConfig, SizeConfig,
    ToleranceConfig, FormatConfig
)


def apply_householder(Q, R, i):
    """Применяет преобразование Хаусхолдера к Q и R на шаге i"""
    m = Q.shape[0]
    x = R[i:, i]
    e = np.zeros_like(x)
    e[0] = np.linalg.norm(x)
    u = x - e

    # Используем EPSILON из конфигурации
    if np.linalg.norm(u) < GeneralConfig.EPSILON:
        return Q, R

    v = u / np.linalg.norm(u)
    H_i = np.eye(m)
    H_i[i:, i:] -= 2.0 * np.outer(v, v)

    R = H_i @ R
    Q = Q @ H_i.T
    return Q, R


def householder_qr(A):
    """QR-разложение методом Хаусхолдера"""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(n):
        Q, R = apply_householder(Q, R, i)
    return Q, R


def print_matrix_info(Q, R):
    """Выводит информацию о матрицах Q и R"""
    print(f"\nQ:\n", np.round(Q, GeneralConfig.MATRIX_PRECISION))
    print(f"R:\n", np.round(R, GeneralConfig.MATRIX_PRECISION))
    print(f"Проверка Q@R:\n", np.round(Q @ R, GeneralConfig.MATRIX_PRECISION))


def print_column_norms(A, R):
    """Выводит нормы второго столбца"""
    col2_A = A[:, 1]
    col2_R = R[:, 1]
    norm_col2_A = np.linalg.norm(col2_A)
    norm_col2_R = np.linalg.norm(col2_R)

    print(f"\nНорма второго столбца в A: {norm_col2_A:.{GeneralConfig.VECTOR_PRECISION}f}")
    print(f"Норма второго столбца в R: {norm_col2_R:.{GeneralConfig.VECTOR_PRECISION}f}")


def run_task1():
    """Выполнение задания 1"""
    # Генерация матрицы с использованием конфигурации
    A = np.random.randint(
        SizeConfig.TASK1_RANDOM_MIN,
        SizeConfig.TASK1_RANDOM_MAX,
        (SizeConfig.TASK1_MATRIX_ROWS, SizeConfig.TASK1_MATRIX_COLS)
    ).astype(float)

    print("Исходная матрица A:\n", A)

    # QR-разложение
    Q, R = householder_qr(A)

    print_matrix_info(Q, R)
    print_column_norms(A, R)

    return Q, R