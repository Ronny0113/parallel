import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

def worker_sort(array):
    """Функция для сортировки подмассива."""
    return np.sort(array)

def parallel_sort(array, num_workers):
    """Функция для параллельной сортировки."""
    # Разделяем массив на подмассивы для каждого рабочего процесса
    split_arrays = np.array_split(array, num_workers)

    with mp.Pool(num_workers) as pool:
        sorted_arrays = pool.map(worker_sort, split_arrays)

    # Объединяем отсортированные подмассивы
    return np.concatenate(sorted_arrays)

def main():
    # Параметры
    num_elements = 70000000
    num_trials = 8

    # Генерируем случайный массив один раз
    random_array = np.random.uniform(0, 10, num_elements)

    # Массив для хранения времени выполнения
    execution_times = []
    ideal_times = []

    for num_workers in range(1, num_trials + 1):
        # Измеряем время выполнения
        start_time = time.time()
        parallel_sort(random_array, num_workers)
        end_time = time.time()

        # Сохраняем время выполнения
        execution_time = end_time - start_time
        execution_times.append(execution_time)

        # Для идеального времени
        ideal_times.append(execution_times[0] / num_workers)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_trials + 1), execution_times, label='Время выполнения', marker='o')
    plt.plot(range(1, num_trials + 1), ideal_times, label='Идеальное время', marker='x', linestyle='--')
    plt.xlabel('Количество процессов')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение времени выполнения сортировки')
    plt.xticks(range(1, num_trials + 1))
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
