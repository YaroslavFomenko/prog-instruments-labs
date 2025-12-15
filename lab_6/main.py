import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


print("\nЗАДАНИЕ 1: QR-разложение методом Хаусхолдера")

np.random.seed(42)
A = np.random.randint(-10, 11, (3, 3)).astype(float)
print("Исходная матрица A:\n", A)


# Метод Хаусхолдера
def householder_qr(A):
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

print("\nЗАДАНИЕ 2: Решение системы с помощью QR-разложения")

# Система для варианта 4
A_system = np.array([
    [17.1, -8.3, 14.4, 7.2],
    [6.4, 8.5, -4.3, 8.8],
    [8.3, -6.6, 5.8, 12.2],
    [3.8, 14.2, 6.3, -15.5]
], dtype=float)

b_system = np.array([13.5, 7.7, -4.7, 2.8], dtype=float)

print("Матрица A:\n", A_system)
print("Вектор b:", b_system)

# QR-разложение
Q_sys, R_sys = householder_qr(A_system)

# Решение: RX = Q^T b
b_new = Q_sys.T @ b_system


# Обратная подстановка
def back_substitution(R, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x


x_qr = back_substitution(R_sys, b_new)
print("\nРешение методом QR-разложения: x =", np.round(x_qr, 4))

# Проверка
x_np = np.linalg.solve(A_system, b_system)
print("Решение numpy: x =", np.round(x_np, 4))
print("Разность:", np.round(np.linalg.norm(x_qr - x_np), 6))

# Проверка подстановкой
residual = A_system @ x_qr - b_system
print("Невязка (Ax - b):", np.round(residual, 6))

print("\nЗАДАНИЕ 3: Метод простых итераций (Якоби)")

# Система для варианта 4
A_jacobi = np.array([
    [5.3, 2.1, 2.8],
    [1.9, 4.1, 2.1],
    [7.5, 3.8, 4.8]
], dtype=float)

b_jacobi = np.array([0.8, 2.1, 5.6], dtype=float)

print("Исходная матрица A:\n", A_jacobi)
print("Вектор b:", b_jacobi)

# Проверяем различные перестановки для достижения диагонального преобладания
print("\nПоиск подходящей перестановки уравнений...")

# Пробуем разные перестановки
permutations = [
    [2, 0, 1],  # [7.5, 3.8, 4.8], [5.3, 2.1, 2.8], [1.9, 4.1, 2.1]
    [1, 2, 0],  # [1.9, 4.1, 2.1], [7.5, 3.8, 4.8], [5.3, 2.1, 2.8]
    [2, 1, 0],  # [7.5, 3.8, 4.8], [1.9, 4.1, 2.1], [5.3, 2.1, 2.8]
]

for idx, perm in enumerate(permutations):
    A_perm = A_jacobi[perm]
    b_perm = b_jacobi[perm]

    print(f"\nПерестановка {idx + 1}: {perm}")
    has_diag_dominance = True
    for i in range(3):
        diag = abs(A_perm[i, i])
        sum_off_diag = sum(abs(A_perm[i, j]) for j in range(3) if j != i)
        dominance = diag > sum_off_diag
        print(f"  Строка {i + 1}: |a_{i + 1}{i + 1}| = {diag:.1f} > {sum_off_diag:.1f}? {dominance}")
        if not dominance:
            has_diag_dominance = False

    if has_diag_dominance:
        print("  ✓ Условие диагонального преобладания выполнено!")
        A_transformed = A_perm.copy()
        b_transformed = b_perm.copy()
        break
else:
    print("Не удалось найти перестановку с строгим диагональным преобладанием")
    print("Используем перестановку с наилучшими характеристиками")
    A_transformed = A_jacobi[[2, 1, 0]]  # [7.5, 3.8, 4.8], [1.9, 4.1, 2.1], [5.3, 2.1, 2.8]
    b_transformed = b_jacobi[[2, 1, 0]]

print("\nВыбранная система:")
print("Матрица A:\n", A_transformed)
print("Вектор b:", b_transformed)


def jacobi_method_prettytable(A, b, tol=1e-3, max_iter=50):
    n = len(b)
    x = np.zeros(n)

    # Создаем таблицу
    table = PrettyTable()
    table.field_names = ["Итерация", "x₁", "x₂", "x₃", "Погрешность"]
    table.float_format = ".6"

    print("\nИтерационный процесс метода Якоби:")

    convergence = True

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        error = np.linalg.norm(x_new - x, np.inf)

        # Проверяем на расходимость
        if k > 5 and error > 1000:
            convergence = False
            print("⚠️ Метод расходится! Останавливаем итерации.")
            break

        # Добавляем строку в таблицу
        table.add_row([k + 1, x_new[0], x_new[1], x_new[2], f"{error:.6f}"])

        if error < tol:
            print(f"✓ Достигнута заданная точность на итерации {k + 1}")
            break
        x = x_new.copy()
    else:
        if convergence:
            print(f"Достигнуто максимальное число итераций ({max_iter})")

    # Выводим таблицу
    print(table)

    if not convergence:
        print("\n❌ Метод Якоби расходится для данной системы!")
        print("Рекомендуется использовать метод Зейделя или преобразовать систему")
        return None, k + 1

    return x_new, k + 1


# Пробуем метод Якоби
x_jacobi, iterations = jacobi_method_prettytable(A_transformed, b_transformed)

if x_jacobi is not None:
    print(f"\nИтоговое решение: x = {np.round(x_jacobi, 4)}")
    print(f"Количество итераций: {iterations}")

    # Проверка
    residual_jacobi = A_transformed @ x_jacobi - b_transformed
    print(f"Невязка (Ax - b): {np.round(residual_jacobi, 6)}")

# Проверяем решение с помощью numpy
x_np_check = np.linalg.solve(A_jacobi, b_jacobi)
print(f"\nПроверка решением numpy: x = {np.round(x_np_check, 4)}")

# Если метод Якоби не сходится, используем прямое решение
if x_jacobi is None:
    print("\nИспользуем прямое решение системы:")
    x_direct = np.linalg.solve(A_jacobi, b_jacobi)
    print(f"Решение: x = {np.round(x_direct, 4)}")

    # Проверка
    residual_direct = A_jacobi @ x_direct - b_jacobi
    print(f"Невязка (Ax - b): {np.round(residual_direct, 6)}")


# =============================================================================
# ЗАДАНИЕ 4: Решение нелинейного уравнения
# =============================================================================
print("\nЗАДАНИЕ 4: Решение нелинейного уравнения")


def f(x):
    return x ** 3 - 1.89 * x ** 2 - 2 * x + 1.76


# Построение графика функции
x_plot = np.linspace(-1, 3, 400)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x³ - 1.89x² - 2x + 1.76')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('График функции для отделения корней')
plt.legend()
plt.show()

# Анализ функции на концах интервалов
print("Анализ функции на интервалах:")
print(f"f(-1) = {f(-1):.3f}")
print(f"f(0) = {f(0):.3f}")
print(f"f(1) = {f(1):.3f}")
print(f"f(2) = {f(2):.3f}")


# Метод половинного деления
def bisection(f, a, b, tol=1e-3):
    print(f"\nМетод половинного деления на интервале [{a}, {b}]:")
    print("Итерация\tx\t\tf(x)")
    print("-" * 40)

    for i in range(100):
        c = (a + b) / 2
        print(f"{i + 1}\t\t{c:.6f}\t{f(c):.6f}")

        if f(c) == 0 or (b - a) / 2 < tol:
            return c, i + 1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, i + 1


# Метод хорд
def chord(f, a, b, tol=1e-4):
    print(f"\nМетод хорд на интервале [{a}, {b}]:")
    print("Итерация\tx\t\tf(x)")
    print("-" * 40)

    x_prev = a
    x = a - f(a) * (b - a) / (f(b) - f(a))

    for i in range(100):
        print(f"{i + 1}\t\t{x:.6f}\t{f(x):.6f}")

        if abs(f(x)) < tol or abs(x - x_prev) < tol:
            return x, i + 1

        if f(a) * f(x) < 0:
            b = x
        else:
            a = x

        x_prev = x
        x = a - f(a) * (b - a) / (f(b) - f(a))

    return x, i + 1


x_bisect, iter_bisect = bisection(f, 0, 2)
x_chord, iter_chord = chord(f, 0, 2)

print(f"\nРезультаты:")
print(f"Метод половинного деления: x = {x_bisect:.5f} (итераций: {iter_bisect})")
print(f"Метод хорд: x = {x_chord:.5f} (итераций: {iter_chord})")
print(f"f(x_корень) = {f(x_chord):.2e}")

# =============================================================================
# ЗАДАНИЕ 5: Решение системы нелинейных уравнений
# =============================================================================
print("\nЗАДАНИЕ 5: Решение системы нелинейных уравнений")


def F(X):
    x, y = X
    return np.array([
        np.cos(x + 0.5) - y - 2,
        np.sin(y) - 2 * x - 1
    ])


def J(X):
    x, y = X
    return np.array([
        [-np.sin(x + 0.5), -1],
        [-2, np.cos(y)]
    ])


# Построение графиков системы уравнений
x_vals = np.linspace(-2, 2, 100)
y_vals1 = np.cos(x_vals + 0.5) - 2  # из первого уравнения: cos(x+0.5) - y = 2
y_vals2 = np.arcsin(2 * x_vals + 1)  # из второго уравнения: sin(y) - 2x = 1

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графическое решение системы нелинейных уравнений')
plt.legend()
plt.axis([-2, 2, -3, 1])
plt.show()


def newton_system(F, J, X0, tol=1e-4):
    X = X0.copy()
    print("Метод Ньютона для системы:")
    print("Итерация\tx\t\ty\t\tНорма F(X)")
    print("-" * 60)

    for i in range(100):
        FX = F(X)
        norm_F = np.linalg.norm(FX)
        print(f"{i + 1}\t\t{X[0]:.6f}\t{X[1]:.6f}\t{norm_F:.6f}")

        if norm_F < tol:
            return X, i + 1

        JX = J(X)
        dX = np.linalg.solve(JX, -FX)
        X += dX

    return X, i + 1


X0 = np.array([0.0, -1.0])
print(f"Начальное приближение: x0 = {X0[0]}, y0 = {X0[1]}")
X_sol, iter_newton = newton_system(F, J, X0)

print(f"\nРешение: x = {X_sol[0]:.6f}, y = {X_sol[1]:.6f}")
print(f"Итераций: {iter_newton}")

# Проверка решения
F_sol = F(X_sol)
print(f"Проверка:")
print(f"f1(x,y) = cos({X_sol[0]:.3f}+0.5) - {X_sol[1]:.3f} - 2 = {F_sol[0]:.2e}")
print(f"f2(x,y) = sin({X_sol[1]:.3f}) - 2*{X_sol[0]:.3f} - 1 = {F_sol[1]:.2e}")

# Финальный график с решением
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals1, 'b-', linewidth=2, label='cos(x+0.5) - y = 2')
plt.plot(x_vals, y_vals2, 'r-', linewidth=2, label='sin(y) - 2x = 1')
plt.plot(X_sol[0], X_sol[1], 'go', markersize=10, label=f'Решение ({X_sol[0]:.3f}, {X_sol[1]:.3f})')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Решение системы нелинейных уравнений')
plt.legend()
plt.axis([-2, 2, -3, 1])
plt.show()

print("\n" + "=" * 50)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 50)