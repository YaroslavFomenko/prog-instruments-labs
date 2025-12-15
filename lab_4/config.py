import numpy as np
from typing import List, Tuple


class GeneralConfig:
    """Общие настройки для всех заданий"""
    RANDOM_SEED: int = 42  # Для воспроизводимости
    MATRIX_PRECISION: int = 4  # Точность вывода матриц
    VECTOR_PRECISION: int = 6  # Точность вывода векторов
    PLOT_FIGSIZE: Tuple[int, int] = (10, 6)  # Размер графиков
    EPSILON: float = 1e-12  # Машинный ноль для сравнений


class ToleranceConfig:
    """Настройки точности для разных методов"""

    # Общие настройки точности
    DEFAULT: float = 1e-3  # Стандартная точность
    HIGH: float = 1e-4  # Высокая точность
    VERY_HIGH: float = 1e-6  # Очень высокая точность
    EXTREME: float = 1e-9  # Максимальная точность

    # Специфичные настройки
    QR_DECOMPOSITION: float = 1e-12  # Для QR-разложения
    JACOBI: float = DEFAULT  # Метод Якоби
    BISECTION: float = DEFAULT  # Метод половинного деления
    CHORD: float = HIGH  # Метод хорд
    NEWTON: float = HIGH  # Метод Ньютона


class IterationConfig:
    """Настройки максимального числа итераций"""

    DEFAULT_MAX: int = 50  # Стандартный лимит
    JACOBI: int = DEFAULT_MAX  # Метод Якоби
    BISECTION: int = 100  # Метод половинного деления
    CHORD: int = 100  # Метод хорд
    NEWTON: int = 100  # Метод Ньютона

    # Проверка на расходимость (Якоби)
    JACOBI_DIVERGENCE_CHECK_START: int = 5  # Когда начинать проверку
    JACOBI_DIVERGENCE_THRESHOLD: float = 1000.0  # Порог расходимости



class SizeConfig:
    """Настройки размеров и параметров"""

    # Размеры матриц для задания 1
    TASK1_MATRIX_ROWS: int = 3
    TASK1_MATRIX_COLS: int = 3
    TASK1_RANDOM_MIN: int = -10
    TASK1_RANDOM_MAX: int = 11

    # Графики
    PLOT_POINTS: int = 400  # Количество точек для графиков
    X_RANGE_NONLINEAR: Tuple[float, float] = (-1, 3)  # Для нелинейного уравнения
    X_RANGE_SYSTEM: Tuple[float, float] = (-2, 2)  # Для системы уравнений
    Y_RANGE_SYSTEM: Tuple[float, float] = (-3, 1)  # Для системы уравнений


class SystemConfig:
    """Конфигурация систем уравнений для варианта 4"""

    @staticmethod
    def get_task2_system() -> Tuple[np.ndarray, np.ndarray]:
        """Система для задания 2"""
        A = np.array([
            [17.1, -8.3, 14.4, 7.2],
            [6.4, 8.5, -4.3, 8.8],
            [8.3, -6.6, 5.8, 12.2],
            [3.8, 14.2, 6.3, -15.5]
        ], dtype=float)

        b = np.array([13.5, 7.7, -4.7, 2.8], dtype=float)
        return A, b

    @staticmethod
    def get_task3_system() -> Tuple[np.ndarray, np.ndarray]:
        """Система для задания 3 (Якоби)"""
        A = np.array([
            [5.3, 2.1, 2.8],
            [1.9, 4.1, 2.1],
            [7.5, 3.8, 4.8]
        ], dtype=float)

        b = np.array([0.8, 2.1, 5.6], dtype=float)
        return A, b

    @staticmethod
    def get_task3_permutations() -> List[List[int]]:
        """Перестановки для поиска диагонального преобладания"""
        return [
            [2, 0, 1],  # [7.5, 3.8, 4.8], [5.3, 2.1, 2.8], [1.9, 4.1, 2.1]
            [1, 2, 0],  # [1.9, 4.1, 2.1], [7.5, 3.8, 4.8], [5.3, 2.1, 2.8]
            [2, 1, 0],  # [7.5, 3.8, 4.8], [1.9, 4.1, 2.1], [5.3, 2.1, 2.8]
        ]

    @staticmethod
    def get_task5_initial_guess() -> np.ndarray:
        """Начальное приближение для системы нелинейных уравнений"""
        return np.array([0.0, -1.0])


class FunctionConfig:
    """Конфигурация функций для нелинейных уравнений"""

    @staticmethod
    def nonlinear_function(x: float) -> float:
        """Функция для задания 4: f(x) = x³ - 1.89x² - 2x + 1.76"""
        return x ** 3 - 1.89 * x ** 2 - 2 * x + 1.76

    @staticmethod
    def nonlinear_system(X: np.ndarray) -> np.ndarray:
        """Система нелинейных уравнений для задания 5"""
        x, y = X
        return np.array([
            np.cos(x + 0.5) - y - 2,
            np.sin(y) - 2 * x - 1
        ])

    @staticmethod
    def nonlinear_system_jacobian(X: np.ndarray) -> np.ndarray:
        """Якобиан системы нелинейных уравнений"""
        x, y = X
        return np.array([
            [-np.sin(x + 0.5), -1],
            [-2, np.cos(y)]
        ])

    @staticmethod
    def get_root_interval() -> Tuple[float, float]:
        """Интервал для поиска корня нелинейного уравнения"""
        return (0.0, 2.0)


class FormatConfig:
    """Настройки форматирования вывода"""

    SEPARATOR_WIDTH: int = 60
    TABLE_ROW_FORMAT: str = ".6f"
    ITERATION_TABLE_HEADERS: List[str] = ["Итерация", "x₁", "x₂", "x₃", "Погрешность"]
    NEWTON_TABLE_HEADERS: List[str] = ["Итерация", "x", "y", "Норма F(X)"]

    @staticmethod
    def create_separator(title: str = "") -> str:
        """Создаёт разделитель с заголовком"""
        if title:
            return f"\n{'=' * FormatConfig.SEPARATOR_WIDTH}\n{title.center(FormatConfig.SEPARATOR_WIDTH)}\n{'=' * FormatConfig.SEPARATOR_WIDTH}"
        return "\n" + "=" * FormatConfig.SEPARATOR_WIDTH