import numpy as np
from task1 import householder_qr
from config import SystemConfig
from utils import print_system, print_residual, compare_with_numpy


def back_substitution(R, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x


def solve_with_qr(A, b):
    Q, R = householder_qr(A)
    b_new = Q.T @ b
    x = back_substitution(R, b_new)
    return x, Q, R


def run_task2():
    A, b = SystemConfig.get_task2_system()
    print_system(A, b)

    x_qr, Q, R = solve_with_qr(A, b)
    print_residual(A, x_qr, b, "QR-разложение")

    compare_with_numpy(A, b, x_qr, "QR-метод")

    return x_qr