import matplotlib.pyplot as plt
import time
import multiprocessing


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


# Функция для работы в отдельном процессе
def process_integrate_f(args):
    a, b, n = args
    return integrate_f(a, b, n)


# Многопроцессорная интеграция
def parallel_integrate_f(a, b, n, num_processes):
    dx = (b - a) / n
    chunk_size = n // num_processes
    args = []
    for i in range(num_processes):
        start_a = a + i * chunk_size * dx  # Начало интервала для текущего процесса
        end_b = a + (i + 1) * chunk_size * dx  # Конец интервала для текущего процесса
        args.append((start_a, end_b, chunk_size))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_integrate_f, args)

    return sum(results)


if __name__ == '__main__':
    # Основные параметры
    a = 5
    b = 10
    n = 10000000
    max_processes = 12  # Максимальное количество процессов

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

    print("Результат:", result)
    print("Идеальное время:", time_ideal)
    print("Реальное время:", time_real)

    # Построение графиков
    plt.plot(range(1, max_processes + 1), time_ideal, label='Идеальная скорость', linestyle='--')
    plt.plot(range(1, max_processes + 1), time_real, label='Реальная скорость', marker='o')
    plt.xlabel('Количество процессов')
    plt.ylabel('Время выполнения (с)')
    plt.title('Сравнение скорости интеграции с использованием процессов')
    plt.legend()
    plt.grid()
    plt.show()
