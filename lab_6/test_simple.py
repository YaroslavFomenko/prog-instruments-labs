import numpy as np
from main import (
    householder_qr,
)


def test_householder_qr_shape():
    """Тест 1: Проверяем что Q и R имеют правильные размеры"""
    np.random.seed(42)
    A = np.random.randint(-10, 11, (3, 3)).astype(float)

    Q, R = householder_qr(A)

    # Проверяем размеры
    assert Q.shape == (3, 3), f"Q должна быть 3x3, получено {Q.shape}"
    assert R.shape == (3, 3), f"R должна быть 3x3, получено {R.shape}"


def test_householder_qr_orthogonality():
    """Тест 2: Проверяем что Q ортогональная (Q^T Q = I)"""
    A = np.array([
        [17.1, -8.3, 14.4, 7.2],
        [6.4, 8.5, -4.3, 8.8],
        [8.3, -6.6, 5.8, 12.2],
        [3.8, 14.2, 6.3, -15.5]
    ], dtype=float)

    Q, _ = householder_qr(A)

    # Q^T * Q должна быть близка к единичной матрице
    QT_Q = np.dot(Q.T, Q)
    identity_4x4 = np.eye(4)

    assert np.allclose(QT_Q, identity_4x4, atol=1e-10), "Q не ортогональна!"


def test_householder_qr_decomposition():
    """Тест 3: Проверяем что A корректно восстанавливается из Q и R"""
    A = np.array([
        [5.3, 2.1, 2.8],
        [1.9, 4.1, 2.1],
        [7.5, 3.8, 4.8]
    ], dtype=float)

    Q, R = householder_qr(A)
    A_reconstructed = np.dot(Q, R)

    # Восстановленная матрица должна совпадать с исходной
    assert np.allclose(A, A_reconstructed, atol=1e-10), \
        "A != Q*R, декомпозиция некорректна!"