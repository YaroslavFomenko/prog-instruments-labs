import numpy as np
from config import GeneralConfig, FormatConfig
from task1 import run_task1
from task2 import run_task2
from task3 import run_task3
from task4 import run_task4
from task5 import run_task5


def main():
    """Запуск всех заданий лабораторной работы"""
    print(FormatConfig.create_separator("ЛАБОРАТОРНАЯ РАБОТА 4: РЕФАКТОРИНГ"))

    # Устанавливаем seed для воспроизводимости
    np.random.seed(GeneralConfig.RANDOM_SEED)

    # Запускаем все задания последовательно
    print(FormatConfig.create_separator("ЗАДАНИЕ 1: QR-разложение методом Хаусхолдера"))
    run_task1()

    print(FormatConfig.create_separator("ЗАДАНИЕ 2: Решение системы с помощью QR-разложения"))
    run_task2()

    print(FormatConfig.create_separator("ЗАДАНИЕ 3: Метод простых итераций (Якоби)"))
    run_task3()

    print(FormatConfig.create_separator("ЗАДАНИЕ 4: Решение нелинейного уравнения"))
    run_task4()

    print(FormatConfig.create_separator("ЗАДАНИЕ 5: Решение системы нелинейных уравнений"))
    run_task5()

    print(FormatConfig.create_separator("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!"))


if __name__ == "__main__":
    main()