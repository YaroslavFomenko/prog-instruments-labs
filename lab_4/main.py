"""
Рефакторинг: Разделение ответственности (Single Responsibility Principle)
"""

import numpy as np
from task1 import run_task1
from task2 import run_task2
from task3 import run_task3
from task4 import run_task4
from task5 import run_task5


def main():
    """Запуск всех заданий лабораторной работы"""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 4: РЕФАКТОРИНГ")
    print("=" * 60)

    # Устанавливаем seed для воспроизводимости
    np.random.seed(42)

    # Запускаем все задания последовательно
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 1: QR-разложение методом Хаусхолдера")
    print("=" * 60)
    run_task1()

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 2: Решение системы с помощью QR-разложения")
    print("=" * 60)
    run_task2()

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 3: Метод простых итераций (Якоби)")
    print("=" * 60)
    run_task3()

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 4: Решение нелинейного уравнения")
    print("=" * 60)
    run_task4()

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 5: Решение системы нелинейных уравнений")
    print("=" * 60)
    run_task5()

    print("\n" + "=" * 60)
    print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
    print("=" * 60)


if __name__ == "__main__":
    main()