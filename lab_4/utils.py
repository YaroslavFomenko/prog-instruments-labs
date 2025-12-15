import numpy as np
from config import GeneralConfig


# 1. Функции вывода матриц/векторов (самое частое дублирование)
def print_matrix(A, name="Матрица"):
    """Выводит матрицу с заданным именем"""
    print(f"\n{name}:")
    print(np.round(A, GeneralConfig.MATRIX_PRECISION))


def print_vector(b, name="Вектор"):
    """Выводит вектор с заданным именем"""
    print(f"{name}: {np.round(b, GeneralConfig.VECTOR_PRECISION)}")


def print_system(A, b):
    """Выводит систему уравнений"""
    print_matrix(A, "Матрица A")
    print_vector(b, "Вектор b")


# 2. Вычисление и вывод норм
def print_norm(x, name="Норма"):
    """Выводит норму вектора"""
    norm_val = np.linalg.norm(x)
    print(f"{name}: {norm_val:.{GeneralConfig.VECTOR_PRECISION}f}")


def print_residual(A, x, b, method_name="Решение"):
    """Выводит невязку решения"""
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)
    print(f"\n{method_name}:")
    print(f"  x = {np.round(x, GeneralConfig.VECTOR_PRECISION)}")
    print(f"  Норма невязки: {residual_norm:.{GeneralConfig.VECTOR_PRECISION}e}")


# 3. Сравнение с numpy (используется в нескольких местах)
def compare_with_numpy(A, b, x_custom, custom_name="Наш метод"):
    """Сравнивает решение с numpy"""
    x_np = np.linalg.solve(A, b)
    diff = np.linalg.norm(x_custom - x_np)

    print(f"\nСравнение с numpy:")
    print(f"  {custom_name}: {np.round(x_custom, GeneralConfig.VECTOR_PRECISION)}")
    print(f"  numpy: {np.round(x_np, GeneralConfig.VECTOR_PRECISION)}")
    print(f"  Разница: {diff:.{GeneralConfig.VECTOR_PRECISION}e}")
    return x_np


# 4. Проверка диагонального преобладания
def is_diagonally_dominant(A):
    """Проверяет, является ли матрица диагонально доминирующей"""
    for i in range(len(A)):
        diag = abs(A[i, i])
        off_diag = sum(abs(A[i, j]) for j in range(len(A)) if j != i)
        if diag <= off_diag:
            return False, i, diag, off_diag
    return True, -1, 0, 0


# 5. Форматирование итераций
def print_iteration_header(method, a=None, b=None):
    """Выводит заголовок для итерационного метода"""
    if a is not None and b is not None:
        print(f"\n{method} на [{a}, {b}]:")
    else:
        print(f"\n{method}:")
    print("Итерация\tx\t\tf(x)")
    print("-" * 50)


def print_iteration(iter_num, x, fx=None):
    """Выводит одну итерацию"""
    if isinstance(x, np.ndarray) and len(x.shape) == 1:
        # Это вектор
        x_str = "\t".join(f"{xi:.6f}" for xi in x)
        if fx is not None:
            print(f"{iter_num}\t\t{x_str}\t{fx:.6f}")
        else:
            print(f"{iter_num}\t\t{x_str}")
    else:
        if fx is not None:
            print(f"{iter_num}\t\t{x:.6f}\t{fx:.6f}")
        else:
            print(f"{iter_num}\t\t{x:.6f}")