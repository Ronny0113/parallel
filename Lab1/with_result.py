import matplotlib.pyplot as plt
import time
import multiprocessing


# Определяем функцию f(x)
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


# Функция для аналитического решения интеграла
def analytical_solution(a, b):
    # Вводите своё аналитическое решение для данной функции
    return -(1 / b) + (1 / a) + (1 / (b ** 2) - 1 / (a ** 2)) + 0.5 * (1 / (b ** 3) - 1 / (a ** 3))


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
    n = 10000000  # Увеличьте количество разбиений для уменьшения накладных расходов
    max_processes = 8  # Максимальное количество процессов

    # Сбор данных о времени выполнения и результата интеграции
    time_real = []
    numerical_results = []
    analytical_results = []

    for num_processes in range(1, max_processes + 1):
        start = time.time()
        result = parallel_integrate_f(a, b, n, num_processes)
        end = time.time()

        time_real.append(end - start)
        numerical_results.append(result)
        analytical_results.append(analytical_solution(a, b))

        # Вывод результатов
        print(f'Количество процессов: {num_processes}, '
              f'Численный интеграл: {result:.6f}, '
              f'Aналитический интеграл: {analytical_results[-1]:.6f}, '
              f'Время выполнения: {time_real[-1]:.6f} секунд')

    # Построение графиков
    plt.plot(range(1, max_processes + 1), numerical_results, label='Численный интеграл', marker='o')
    plt.plot(range(1, max_processes + 1), analytical_results, label='Аналитический интеграл', linestyle='--')
    plt.xlabel('Количество процессов')
    plt.ylabel('Значение интеграла')
    plt.title('Сравнение численного и аналитического интегралов')
    plt.legend()
    plt.grid()
    plt.show()
