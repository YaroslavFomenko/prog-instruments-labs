import numpy as np
from prettytable import PrettyTable
from config import SystemConfig, ToleranceConfig, IterationConfig
from utils import print_system, is_diagonally_dominant, print_residual, compare_with_numpy


def find_best_permutation(A, b):
    permutations = SystemConfig.get_task3_permutations()

    print("Поиск перестановки с диагональным преобладанием...")

    for idx, perm in enumerate(permutations):
        A_perm = A[perm]
        b_perm = b[perm]

        print(f"\nПерестановка {idx + 1}: {perm}")
        is_dominant, row, diag, off_diag = is_diagonally_dominant(A_perm)

        for i in range(len(A_perm)):
            diag_i = abs(A_perm[i, i])
            off_diag_i = sum(abs(A_perm[i, j]) for j in range(len(A_perm)) if j != i)
            dominant_i = diag_i > off_diag_i
            print(f"  Строка {i + 1}: |a_{i + 1}{i + 1}| = {diag_i:.1f} > {off_diag_i:.1f}? {dominant_i}")

        if is_dominant:
            print("  ✓ Диагональное преобладание!")
            return A_perm, b_perm

    print("Не нашли строгого преобладания, используем первую перестановку")
    return A[permutations[0]], b[permutations[0]]


def jacobi_method(A, b):
    n = len(b)
    x = np.zeros(n)

    table = PrettyTable()
    table.field_names = ["Итерация", "x₁", "x₂", "x₃", "Погрешность"]
    table.float_format = ".6"

    print("\nМетод Якоби:")

    for k in range(IterationConfig.JACOBI):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        error = np.linalg.norm(x_new - x, np.inf)
        table.add_row([k + 1, x_new[0], x_new[1], x_new[2], f"{error:.6f}"])

        if k > IterationConfig.JACOBI_DIVERGENCE_CHECK_START and error > IterationConfig.JACOBI_DIVERGENCE_THRESHOLD:
            print("⚠️ Расходится!")
            return None, k + 1

        if error < ToleranceConfig.JACOBI:
            print(f"✓ Сошёлся на итерации {k + 1}")
            print(table)
            return x_new, k + 1

        x = x_new.copy()

    print(f"Достигнут лимит ({IterationConfig.JACOBI} итераций)")
    print(table)
    return x, IterationConfig.JACOBI


def run_task3():
    A, b = SystemConfig.get_task3_system()
    print_system(A, b)

    A_perm, b_perm = find_best_permutation(A, b)
    print_system(A_perm, b_perm)

    x_jacobi, iterations = jacobi_method(A_perm, b_perm)

    if x_jacobi is not None:
        print_residual(A_perm, x_jacobi, b_perm, "Метод Якоби")

    compare_with_numpy(A, b, x_jacobi if x_jacobi is not None else np.linalg.solve(A, b), "Якоби")

    return x_jacobi if x_jacobi is not None else np.linalg.solve(A, b)