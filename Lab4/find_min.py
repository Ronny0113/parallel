import numpy as np
import random
import math
import time

# Фиксируем сид для генерации случайных чисел
np.random.seed(113)

# Параметры задачи
n = 50  # Размер матрицы

# Генерация случайной матрицы коэффициентов
b = np.random.randint(-50, 51, size=n)
matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        matrix[i, j] = np.random.randint(-50, 51)


# Функция для расчета энергии E
def calculate_E(config, b, matrix, n):
    E = sum(b[i] * config[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            E += matrix[i, j] * config[i] * config[j]
    return E


def simulated_annealing(b, matrix, n, num_iterations=2000, max_time=60):
    # Инициализация случайной конфигурации
    current_config = [random.randint(-2, 2) for _ in range(n)]
    current_E = calculate_E(current_config, b, matrix, n)

    T = 1000
    min_config = current_config.copy()
    min_E = current_E

    start_time = time.time()

    while time.time() - start_time < max_time:
        for _ in range(num_iterations):
            # Проверяем время каждые несколько итераций, чтобы прервать процесс
            if time.time() - start_time >= max_time:
                break

            # Выбираем случайный индекс и новое случайное значение для этого индекса
            i = random.randint(0, n - 1)
            new_config = current_config.copy()
            new_config[i] = max(-2, min(2, new_config[i] + random.choice([-1, 1])))

            new_E = calculate_E(new_config, b, matrix, n)

            # Сравнение энергий
            if new_E <= current_E:
                current_config = new_config
                current_E = new_E
            else:
                p = math.exp(-(new_E - current_E) / T)
                if random.random() < p:
                    current_config = new_config
                    current_E = new_E

            # Обновляем минимальное значение
            if current_E < min_E:
                min_E = current_E
                min_config = current_config.copy()

        T *= 0.95

    return min_config, min_E


# Генерация и вывод матрицы
print("Матрица коэффициентов b:")
print(b)
print("\nМатрица коэффициентов:")
print(matrix)

# Выполнение программы
if __name__ == '__main__':
    print("Запуск поиска минимума на 5 минут...")
    best_config, best_E = simulated_annealing(b, matrix, n, max_time=60)
    print("\nРезультат после -5 минут работы:")
    print(f"Лучшая конфигурация: {best_config}")
    print(f"Минимальное значение E: {best_E}")
