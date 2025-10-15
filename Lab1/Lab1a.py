import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor


# Определяем функцию
def f(x):
    return (1 / x) - (1 / x ** 2) - (1 / x ** 3)


# Мы должны интегрировать f(x) от a до b
def integrate_f(a, b, n):
    dx = (b - a) / n
    integral = 0.0
    for i in range(n):
        x = a + i * dx
        integral += f(x)
    integral *= dx
    return integral


# Многопоточная интеграция
def parallel_integrate_f(a, b, n, num_processes):
    dx = (b - a) / n
    chunk_size = n // num_processes
    futures = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size if i != num_processes - 1 else n
            futures.append(
                executor.submit(integrate_f, a + start_index * dx, a + end_index * dx, end_index - start_index))

        results = [future.result() for future in futures]

    return sum(results)


if __name__ == '__main__':
    # Основные параметры
    a = 5
    b = 10
    n = 100000  # Увеличьте количество разбиений для уменьшения накладных расходов
    max_processes = 4  # Максимальное количество потоков, соответствующее допустимому количеству ядер

    # Сбор данных о времени выполнения
    time_ideal = []
    time_real = []

    for num_processes in range(1, max_processes + 1):
        start = time.time()
        result = parallel_integrate_f(a, b, n, num_processes)
        end = time.time()

        time_real.append(end - start)
        if num_processes == 1:
            time_ideal.append(end - start)  # Сохраняем время для одного процесса
        else:
            time_ideal.append(time_real[0] / num_processes)  # t/num_processes, где t - время для 1 процесса

    # Построение графиков
    plt.plot(range(1, max_processes + 1), time_ideal, label='Идеальная скорость', linestyle='--')
    plt.plot(range(1, max_processes + 1), time_real, label='Реальная скорость', marker='o')
    plt.yscale('log')  # Логарифмическая шкала для ясности
    plt.xlabel('Количество потоков')
    plt.ylabel('Время выполнения (с)')
    plt.title('Сравнение скорости интеграции с использованием процессов')
    plt.legend()
    plt.grid()
    plt.show()
