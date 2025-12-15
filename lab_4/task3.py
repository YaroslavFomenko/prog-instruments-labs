import numpy as np
from prettytable import PrettyTable


def find_best_permutation(A, b):
    """Поиск перестановки уравнений для достижения диагонального преобладания"""
    n = len(A)

    # Пробуем разные перестановки
    permutations = [
        [2, 0, 1],  # [7.5, 3.8, 4.8], [5.3, 2.1, 2.8], [1.9, 4.1, 2.1]
        [1, 2, 0],  # [1.9, 4.1, 2.1], [7.5, 3.8, 4.8], [5.3, 2.1, 2.8]
        [2, 1, 0],  # [7.5, 3.8, 4.8], [1.9, 4.1, 2.1], [5.3, 2.1, 2.8]
    ]

    print("Поиск подходящей перестановки уравнений...")

    for idx, perm in enumerate(permutations):
        A_perm = A[perm]
        b_perm = b[perm]

        print(f"\nПерестановка {idx + 1}: {perm}")
        has_diag_dominance = True

        for i in range(n):
            diag = abs(A_perm[i, i])
            sum_off_diag = sum(abs(A_perm[i, j]) for j in range(n) if j != i)
            dominance = diag > sum_off_diag
            print(f"  Строка {i + 1}: |a_{i + 1}{i + 1}| = {diag:.1f} > {sum_off_diag:.1f}? {dominance}")
            if not dominance:
                has_diag_dominance = False

        if has_diag_dominance:
            print("  ✓ Условие диагонального преобладания выполнено!")
            return A_perm, b_perm

    print("Не удалось найти перестановку с строгим диагональным преобладанием")
    print("Используем перестановку с наилучшими характеристиками")
    return A[[2, 1, 0]], b[[2, 1, 0]]


def jacobi_method_prettytable(A, b, tol=1e-3, max_iter=50):
    """Метод Якоби с выводом в таблицу"""
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


def run_task3():
    """Выполнение задания 3"""
    # Система для варианта 4
    A = np.array([
        [5.3, 2.1, 2.8],
        [1.9, 4.1, 2.1],
        [7.5, 3.8, 4.8]
    ], dtype=float)

    b = np.array([0.8, 2.1, 5.6], dtype=float)

    print("Исходная матрица A:\n", A)
    print("Вектор b:", b)

    # Поиск перестановки с диагональным преобладанием
    A_transformed, b_transformed = find_best_permutation(A, b)

    print("\nВыбранная система:")
    print("Матрица A:\n", A_transformed)
    print("Вектор b:", b_transformed)

    # Применение метода Якоби
    x_jacobi, iterations = jacobi_method_prettytable(A_transformed, b_transformed)

    if x_jacobi is not None:
        print(f"\nИтоговое решение: x = {np.round(x_jacobi, 4)}")
        print(f"Количество итераций: {iterations}")

        # Проверка
        residual_jacobi = A_transformed @ x_jacobi - b_transformed
        print(f"Невязка (Ax - b): {np.round(residual_jacobi, 6)}")

    # Проверяем решение с помощью numpy
    x_np_check = np.linalg.solve(A, b)
    print(f"\nПроверка решением numpy: x = {np.round(x_np_check, 4)}")

    # Если метод Якоби не сходится, используем прямое решение
    if x_jacobi is None:
        print("\nИспользуем прямое решение системы:")
        x_direct = np.linalg.solve(A, b)
        print(f"Решение: x = {np.round(x_direct, 4)}")

        # Проверка
        residual_direct = A @ x_direct - b
        print(f"Невязка (Ax - b): {np.round(residual_direct, 6)}")

    return x_jacobi if x_jacobi is not None else x_np_check