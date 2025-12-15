import numpy as np
from prettytable import PrettyTable


def check_row_dominance(A, i):
    """Проверяет диагональное преобладание для строки i"""
    diag = abs(A[i, i])
    sum_off_diag = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
    return diag, sum_off_diag, diag > sum_off_diag


def evaluate_permutation(A_perm, b_perm, perm_idx):
    """Оценивает перестановку на диагональное преобладание"""
    print(f"\nПерестановка {perm_idx + 1}: {perm_idx}")
    all_dominant = True

    for i in range(len(A_perm)):
        diag, sum_off, is_dominant = check_row_dominance(A_perm, i)
        print(f"  Строка {i + 1}: |a_{i + 1}{i + 1}| = {diag:.1f} > {sum_off:.1f}? {is_dominant}")
        if not is_dominant:
            all_dominant = False

    return all_dominant


def find_best_permutation(A, b):
    """Поиск перестановки уравнений для достижения диагонального преобладания"""
    permutations = [
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
    ]

    print("Поиск подходящей перестановки уравнений...")

    for idx, perm in enumerate(permutations):
        A_perm = A[perm]
        b_perm = b[perm]

        if evaluate_permutation(A_perm, b_perm, idx):
            print("  ✓ Условие диагонального преобладания выполнено!")
            return A_perm, b_perm

    print("Не удалось найти перестановку с строгим диагональным преобладанием")
    print("Используем перестановку с наилучшими характеристиками")
    return A[[2, 1, 0]], b[[2, 1, 0]]


def compute_jacobi_iteration(A, b, x):
    """Вычисляет одну итерацию метода Якоби"""
    n = len(b)
    x_new = np.zeros(n)

    for i in range(n):
        s = sum(A[i, j] * x[j] for j in range(n) if j != i)
        x_new[i] = (b[i] - s) / A[i, i]

    return x_new


def create_iteration_table():
    """Создаёт таблицу для отображения итераций"""
    table = PrettyTable()
    table.field_names = ["Итерация", "x₁", "x₂", "x₃", "Погрешность"]
    table.float_format = ".6"
    return table


def check_divergence(iteration, error):
    """Проверяет, расходится ли метод"""
    return iteration > 5 and error > 1000


def jacobi_method_prettytable(A, b, tol=1e-3, max_iter=50):
    """Метод Якоби с выводом в таблицу"""
    n = len(b)
    x = np.zeros(n)
    table = create_iteration_table()
    convergence = True

    print("\nИтерационный процесс метода Якоби:")

    for k in range(max_iter):
        x_new = compute_jacobi_iteration(A, b, x)
        error = np.linalg.norm(x_new - x, np.inf)

        if check_divergence(k, error):
            convergence = False
            print("⚠️ Метод расходится! Останавливаем итерации.")
            break

        table.add_row([k + 1, x_new[0], x_new[1], x_new[2], f"{error:.6f}"])

        if error < tol:
            print(f"✓ Достигнута заданная точность на итерации {k + 1}")
            break

        x = x_new.copy()

    if convergence and k == max_iter - 1:
        print(f"Достигнуто максимальное число итераций ({max_iter})")

    print(table)

    if not convergence:
        print("\n❌ Метод Якоби расходится для данной системы!")
        return None, k + 1

    return x_new, k + 1


def print_original_system(A, b):
    """Выводит исходную систему"""
    print("Исходная матрица A:\n", A)
    print("Вектор b:", b)


def print_transformed_system(A_transformed, b_transformed):
    """Выводит преобразованную систему"""
    print("\nВыбранная система:")
    print("Матрица A:\n", A_transformed)
    print("Вектор b:", b_transformed)


def print_jacobi_solution(x_jacobi, iterations, A_transformed, b_transformed):
    """Выводит решение методом Якоби"""
    if x_jacobi is not None:
        print(f"\nИтоговое решение: x = {np.round(x_jacobi, 4)}")
        print(f"Количество итераций: {iterations}")

        # Проверка
        residual_jacobi = A_transformed @ x_jacobi - b_transformed
        print(f"Невязка (Ax - b): {np.round(residual_jacobi, 6)}")


def print_numpy_comparison(A, b, x_jacobi):
    """Сравнивает с решением numpy"""
    x_np_check = np.linalg.solve(A, b)
    print(f"\nПроверка решением numpy: x = {np.round(x_np_check, 4)}")

    # Если метод Якоби не сходится, используем прямое решение
    if x_jacobi is None:
        print("\nИспользуем прямое решение системы:")
        x_direct = x_np_check
        print(f"Решение: x = {np.round(x_direct, 4)}")

        # Проверка
        residual_direct = A @ x_direct - b
        print(f"Невязка (Ax - b): {np.round(residual_direct, 6)}")
        return x_direct

    return x_jacobi


def run_task3():
    """Выполнение задания 3"""
    # Система для варианта 4
    A = np.array([
        [5.3, 2.1, 2.8],
        [1.9, 4.1, 2.1],
        [7.5, 3.8, 4.8]
    ], dtype=float)

    b = np.array([0.8, 2.1, 5.6], dtype=float)

    print_original_system(A, b)

    # Поиск перестановки с диагональным преобладанием
    A_transformed, b_transformed = find_best_permutation(A, b)

    print_transformed_system(A_transformed, b_transformed)

    # Применение метода Якоби
    x_jacobi, iterations = jacobi_method_prettytable(A_transformed, b_transformed)

    print_jacobi_solution(x_jacobi, iterations, A_transformed, b_transformed)

    return print_numpy_comparison(A, b, x_jacobi)