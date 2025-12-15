import numpy as np
from prettytable import PrettyTable
from config import (
    GeneralConfig, SystemConfig,
    ToleranceConfig, IterationConfig,
    FormatConfig
)


def check_row_dominance(A, i):
    """Проверяет диагональное преобладание для строки i"""
    diag = abs(A[i, i])
    sum_off_diag = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
    return diag, sum_off_diag, diag > sum_off_diag


def evaluate_permutation(A_perm, b_perm, perm_idx, perm):
    """Оценивает перестановку на диагональное преобладание"""
    print(f"\nПерестановка {perm_idx + 1}: {perm}")
    all_dominant = True

    for i in range(len(A_perm)):
        diag, sum_off, is_dominant = check_row_dominance(A_perm, i)
        print(f"  Строка {i + 1}: |a_{i + 1}{i + 1}| = {diag:.1f} > {sum_off:.1f}? {is_dominant}")
        if not is_dominant:
            all_dominant = False

    return all_dominant


def find_best_permutation(A, b):
    """Поиск перестановки уравнений для достижения диагонального преобладания"""
    permutations = SystemConfig.get_task3_permutations()

    print("Поиск подходящей перестановки уравнений...")

    for idx, perm in enumerate(permutations):
        A_perm = A[perm]
        b_perm = b[perm]

        if evaluate_permutation(A_perm, b_perm, idx, perm):
            print("  ✓ Условие диагонального преобладания выполнено!")
            return A_perm, b_perm

    print("Не удалось найти перестановку с строгим диагонального преобладания")
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
    table.field_names = FormatConfig.ITERATION_TABLE_HEADERS
    table.float_format = FormatConfig.TABLE_ROW_FORMAT
    return table


def check_divergence(iteration, error):
    """Проверяет, расходится ли метод"""
    return (iteration > IterationConfig.JACOBI_DIVERGENCE_CHECK_START and
            error > IterationConfig.JACOBI_DIVERGENCE_THRESHOLD)


def jacobi_method_prettytable(A, b):
    """Метод Якоби с выводом в таблицу"""
    n = len(b)
    x = np.zeros(n)
    table = create_iteration_table()
    convergence = True

    print("\nИтерационный процесс метода Якоби:")

    for k in range(IterationConfig.JACOBI):
        x_new = compute_jacobi_iteration(A, b, x)
        error = np.linalg.norm(x_new - x, np.inf)

        if check_divergence(k, error):
            convergence = False
            print("⚠️ Метод расходится! Останавливаем итерации.")
            break

        table.add_row([k + 1, x_new[0], x_new[1], x_new[2], f"{error:.{GeneralConfig.VECTOR_PRECISION}f}"])

        if error < ToleranceConfig.JACOBI:
            print(f"✓ Достигнута заданная точность на итерации {k + 1}")
            break

        x = x_new.copy()

    if convergence and k == IterationConfig.JACOBI - 1:
        print(f"Достигнуто максимальное число итераций ({IterationConfig.JACOBI})")

    print(table)

    if not convergence:
        print("\n❌ Метод Якоби расходится для данной системы!")
        return None, k + 1

    return x_new, k + 1


def print_original_system(A, b):
    """Выводит исходную систему"""
    print("Исходная матрица A:\n", np.round(A, GeneralConfig.MATRIX_PRECISION))
    print("Вектор b:", np.round(b, GeneralConfig.VECTOR_PRECISION))


def print_transformed_system(A_transformed, b_transformed):
    """Выводит преобразованную систему"""
    print("\nВыбранная система:")
    print("Матрица A:\n", np.round(A_transformed, GeneralConfig.MATRIX_PRECISION))
    print("Вектор b:", np.round(b_transformed, GeneralConfig.VECTOR_PRECISION))


def print_jacobi_solution(x_jacobi, iterations, A_transformed, b_transformed):
    """Выводит решение методом Якоби"""
    if x_jacobi is not None:
        print(f"\nИтоговое решение: x = {np.round(x_jacobi, GeneralConfig.VECTOR_PRECISION)}")
        print(f"Количество итераций: {iterations}")

        # Проверка
        residual_jacobi = A_transformed @ x_jacobi - b_transformed
        residual_norm = np.linalg.norm(residual_jacobi)
        print(f"Норма невязки (Ax - b): {residual_norm:.{GeneralConfig.VECTOR_PRECISION}f}")


def print_numpy_comparison(A, b, x_jacobi):
    """Сравнивает с решением numpy"""
    x_np_check = np.linalg.solve(A, b)
    print(f"\nПроверка решением numpy: x = {np.round(x_np_check, GeneralConfig.VECTOR_PRECISION)}")

    # Если метод Якоби не сходится, используем прямое решение
    if x_jacobi is None:
        print("\nИспользуем прямое решение системы:")
        x_direct = x_np_check
        print(f"Решение: x = {np.round(x_direct, GeneralConfig.VECTOR_PRECISION)}")

        # Проверка
        residual_direct = A @ x_direct - b
        residual_norm = np.linalg.norm(residual_direct)
        print(f"Норма невязки (Ax - b): {residual_norm:.{GeneralConfig.VECTOR_PRECISION}f}")
        return x_direct

    return x_jacobi


def run_task3():
    """Выполнение задания 3"""
    # Получаем систему из конфигурации
    A, b = SystemConfig.get_task3_system()

    print_original_system(A, b)

    # Поиск перестановки с диагональным преобладанием
    A_transformed, b_transformed = find_best_permutation(A, b)

    print_transformed_system(A_transformed, b_transformed)

    # Применение метода Якоби
    x_jacobi, iterations = jacobi_method_prettytable(A_transformed, b_transformed)

    print_jacobi_solution(x_jacobi, iterations, A_transformed, b_transformed)

    return print_numpy_comparison(A, b, x_jacobi)