import numpy as np
from main import (
    householder_qr,
    back_substitution,
    bisection,
    chord
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


def test_back_substitution():
    """Тест 4: Проверяем алгоритм обратной подстановки"""

    A_system = np.array([
        [17.1, -8.3, 14.4, 7.2],
        [6.4, 8.5, -4.3, 8.8],
        [8.3, -6.6, 5.8, 12.2],
        [3.8, 14.2, 6.3, -15.5]
    ], dtype=float)

    _, R = householder_qr(A_system)

    # Создаем тестовый вектор b
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    x = back_substitution(R, b)

    # Проверяем через numpy (эталон)
    x_expected = np.linalg.solve(R, b)

    assert np.allclose(x, x_expected, atol=1e-10), \
        f"Ошибка обратной подстановки! Получено: {x}, ожидалось: {x_expected}"

    # Дополнительная проверка: R*x должен равняться b
    b_calculated = np.dot(R, x)
    assert np.allclose(b_calculated, b, atol=1e-10), \
        "R*x != b, решение некорректно!"


def test_bisection_linear():
    """Тест 5: Метод половинного деления для функции из задания 4"""

    def f(x):
        return x ** 3 - 1.89 * x ** 2 - 2 * x + 1.76

    root, iterations = bisection(f, 0.0, 2.0, tol=1e-6)

    # Проверяем что нашли корень
    assert abs(f(root)) < 1e-4, \
        f"Найденный корень {root} не удовлетворяет f(x)=0, f({root}) = {f(root)}"

    assert iterations > 0, "Должна быть выполнена хотя бы одна итерация"
    # Корень должен быть между 0 и 2
    assert 0 <= root <= 2, f"Корень {root} должен быть в [0, 2]"


def test_chord_method():
    """Тест 6: Проверяем метод хорд для функции из задания 4"""

    def f(x):
        return x ** 3 - 1.89 * x ** 2 - 2 * x + 1.76

    root, iterations = chord(f, 0.0, 2.0, tol=1e-6)

    # Проверяем результат
    assert abs(f(root)) < 1e-4, \
        f"Метод хорд не нашел корень! f({root}) = {f(root)}"

    assert 0 <= root <= 2, \
        f"Корень должен быть в [0, 2], получено {root}"

    assert iterations > 0, "Метод хорд должен выполнять итерации"
