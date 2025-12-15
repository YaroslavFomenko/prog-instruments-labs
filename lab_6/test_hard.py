import numpy as np
import pytest
from main import (
    householder_qr
)


@pytest.mark.parametrize("matrix_name,matrix_data", [
    ("random_3x3", lambda: np.random.randint(-10, 11, (3, 3)).astype(float)),
    ("system_4x4", lambda: np.array([
        [17.1, -8.3, 14.4, 7.2],
        [6.4, 8.5, -4.3, 8.8],
        [8.3, -6.6, 5.8, 12.2],
        [3.8, 14.2, 6.3, -15.5]
    ], dtype=float)),
    ("jacobi_3x3", lambda: np.array([
        [5.3, 2.1, 2.8],
        [1.9, 4.1, 2.1],
        [7.5, 3.8, 4.8]
    ], dtype=float)),
])
def test_householder_qr_all_matrices_from_code(matrix_name, matrix_data):
    """
    Тест 1: Тестируем QR-разложение для всех матриц из кода
    """
    np.random.seed(42)

    A = matrix_data()
    print(f"\nТестируем QR для матрицы: {matrix_name} ({A.shape[0]}x{A.shape[1]})")

    Q, R = householder_qr(A)

    # Проверка размеров
    n = A.shape[0]
    assert Q.shape == (n, n), f"Q должна быть {n}x{n}"
    assert R.shape == (n, n), f"R должна быть {n}x{n}"

    # Проверка ортогональности Q
    identity = np.eye(n)
    assert np.allclose(Q.T @ Q, identity, atol=1e-10), f"Q не ортогональна для {matrix_name}"

    # Проверка восстановления A
    A_reconstructed = Q @ R
    assert np.allclose(A, A_reconstructed, atol=1e-10), f"Не восстанавливается A для {matrix_name}"

    print(f"  ✓ Все проверки пройдены для {matrix_name}")


