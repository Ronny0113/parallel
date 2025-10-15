import numpy as np
import random
import math
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

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


# Функция для расчета E
def calculate_E(config, b, matrix, n):
    E = sum(b[i] * config[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            E += matrix[i, j] * config[i] * config[j]
    return E


def fmin(b, matrix, n, stop_event, result_queue, num_iterations=2000):
    # Инициализация случайной конфигурации
    current_config = [random.randint(-2, 2) for _ in range(n)]
    current_E = calculate_E(current_config, b, matrix, n)

    min_config = current_config.copy()
    min_E = current_E
    Etalon = -30300.0
    while min_E > Etalon:
        T = 500
        while T > 0.1:
            if stop_event.is_set():  # Проверка, не нашёл ли другой поток решение
                return

            for _ in range(num_iterations):
                if stop_event.is_set():  # Если сигнал остановки
                    return

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
                if min_E <= Etalon:
                    result_queue.put((min_config, min_E))  # Передаём результат через очередь
                    stop_event.set()  # Сигнализируем, что решение найдено

            T *= 0.95


# Функция для многопоточного поиска решения
def parallel_search(num_threads):
    stop_event = mp.Event()  # Событие для остановки потоков
    result_queue = mp.Queue()  # Очередь для передачи результатов
    processes = []

    # Запуск потоков
    for _ in range(num_threads):
        p = mp.Process(target=fmin, args=(b, matrix, n, stop_event, result_queue))
        processes.append(p)
        p.start()

    # Ожидание завершения одного из потоков и получение результата
    best_config, best_E = result_queue.get()

    # Остановка всех потоков
    for p in processes:
        p.join()

    return best_config, best_E


# Запуск многопоточного поиска с замером времени
def run_parallel_search():
    times = []
    thread_counts = list(range(1, 16))
    for num_threads in thread_counts:
        start_time = time.time()
        best_config, best_E = parallel_search(num_threads)
        end_time = time.time()
        times.append(end_time - start_time)
        print(
            f"Потоки: {num_threads}, Время: {end_time - start_time:.4f} сек, Конфигурация: {best_config}, Минимальная E: {best_E}")

    return thread_counts, times


# Выполнение программы
if __name__ == '__main__':
    # Генерация и вывод матрицы
    print("Коэффициенты alpha:")
    print(b)
    print("\nМатрица коэффициентов beta:")
    print(matrix)

    thread_counts, times = run_parallel_search()

    # Построение графика зависимости времени выполнения от числа потоков
    plt.plot(thread_counts, times, marker='o')
    plt.title('Зависимость времени выполнения от количества потоков')
    plt.xlabel('Количество потоков')
    plt.ylabel('Время выполнения (сек)')
    plt.grid(True)
    plt.show()
