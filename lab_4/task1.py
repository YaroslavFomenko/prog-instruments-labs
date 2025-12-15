import numpy as np


def householder_qr(A):
    """QR-разложение методом Хаусхолдера"""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for i in range(n):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        if np.linalg.norm(u) < 1e-12:
            continue
        v = u / np.linalg.norm(u)
        H_i = np.eye(m)
        H_i[i:, i:] -= 2.0 * np.outer(v, v)
        R = H_i @ R
        Q = Q @ H_i.T
    return Q, R


def run_task1():
    """Выполнение задания 1"""
    # Генерация матрицы
    A = np.random.randint(-10, 11, (3, 3)).astype(float)
    print("Исходная матрица A:\n", A)

    # QR-разложение
    Q, R = householder_qr(A)

    print("\nQ:\n", np.round(Q, 4))
    print("R:\n", np.round(R, 4))
    print("Проверка Q@R:\n", np.round(Q @ R, 4))

    # Проверка норм второго столбца
    col2_A = A[:, 1]
    col2_R = R[:, 1]
    norm_col2_A = np.linalg.norm(col2_A)
    norm_col2_R = np.linalg.norm(col2_R)

    print(f"\nНорма второго столбца в A: {norm_col2_A:.6f}")
    print(f"Норма второго столбца в R: {norm_col2_R:.6f}")

    return Q, R