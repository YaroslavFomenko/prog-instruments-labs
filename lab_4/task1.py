import numpy as np
from config import SizeConfig, ToleranceConfig
from utils import print_matrix, print_norm


def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(n):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        if np.linalg.norm(u) < ToleranceConfig.QR_DECOMPOSITION:
            continue
        v = u / np.linalg.norm(u)
        H_i = np.eye(m)
        H_i[i:, i:] -= 2.0 * np.outer(v, v)
        R = H_i @ R
        Q = Q @ H_i.T
    return Q, R


def run_task1():
    A = np.random.randint(
        SizeConfig.TASK1_RANDOM_MIN,
        SizeConfig.TASK1_RANDOM_MAX,
        (SizeConfig.TASK1_MATRIX_ROWS, SizeConfig.TASK1_MATRIX_COLS)
    ).astype(float)

    print_matrix(A, "Исходная матрица A")

    Q, R = householder_qr(A)

    print_matrix(Q, "Матрица Q")
    print_matrix(R, "Матрица R")
    print_matrix(Q @ R, "Проверка Q@R")

    col2_A = A[:, 1]
    col2_R = R[:, 1]

    print("\nНормы второго столбца:")
    print_norm(col2_A, "В A")
    print_norm(col2_R, "В R")

    return Q, R