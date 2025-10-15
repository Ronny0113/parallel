import numpy as np
import itertools
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

# Условия задачи
n = 20
T = 1.0  # Значение T, например, 1.0 - этот параметр может быть заменен на другое число

# Генерация весов агентов
A = np.random.uniform(-10, 10, n)
B = np.random.uniform(-10, 10, (n, n))


# Функция для вычисления E
def calculate_E(configuration):
    E = np.sum(A * configuration)  # Индивидуальные веса
    for i, j in itertools.combinations(range(n), 2):
        E += B[i, j] * configuration[i] * configuration[j]  # Парные веса
    return E


# Функция для параллельного вычисления всех E
def parallel_calculate_E(process_count):
    configurations = list(itertools.product([-1, 1], repeat=n))

    with mp.Pool(process_count) as pool:
        E_values = pool.map(calculate_E, configurations)
    return E_values


# Основная функция для оценки времени
def measure_time(process_count):
    start_time = time.time()
    E_values = parallel_calculate_E(process_count)
    end_time = time.time()

    min_E = np.min(E_values)  # Минимальное E
    avg_E = np.mean(E_values)  # Среднее E

    execution_time = end_time - start_time
    return execution_time, min_E, avg_E, E_values


# Основной блок кода
if __name__ == "__main__":
    # Сбор данных для графика времени
    execution_times = []
    ideal_times = []
    min_E_values = []
    avg_E_values = []
    all_E_values = []

    # Измерение времени выполнения от 1 до 8 процессов
    for process_count in range(1, 9):
        exec_time, min_E, avg_E, E_values = measure_time(process_count)
        execution_times.append(exec_time)
        min_E_values.append(min_E)
        avg_E_values.append(avg_E)
        all_E_values.append(E_values)

        # Определение идеального времени
        if process_count == 1:
            t1 = exec_time
        ideal_times.append(t1 / process_count)

    # Расчет вкладов и нормировка
    contributions = []
    for E_values in all_E_values:
        rho_n = np.exp(-(E_values - min_E_values[-1]) / T)
        contributions.append(rho_n)

    # Нормировка
    rho_percent = []
    for rho_n in contributions:
        z = np.sum(rho_n)
        rho_n_percent = rho_n / z
        rho_percent.append(rho_n_percent)

    # Усреднение вкладов для графика
    avg_rho_percent = np.array([np.mean(rho) for rho in rho_percent])

    # Нахождение самого мощного коэффициента бета
    beta_coefficients = np.array([np.max(np.abs(rho)) for rho in rho_percent])  # Коэффициенты по модулю
    avg_beta = np.mean(beta_coefficients)  # Среднее значение бета

    print("Execution Times:", execution_times)
    print("Ideal Times:", ideal_times)
    print("Min E Values:", min_E_values)
    print("Avg E Values:", avg_E_values)
    print("Average Beta Coefficient:", avg_beta)

    # Вывод вкладов и нормированных вкладов
    for idx, (rho_n, rho_n_percent) in enumerate(zip(contributions, rho_percent), start=1):
        print(f"\nПроцесс {idx}:")
        print("  Вклад (ρ(n)):", rho_n)
        print("  Нормированный вклад (ρ_percent(n)):", rho_n_percent)

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # График времени выполнения
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 9), execution_times, marker='o', label='Реальное время')
    plt.plot(range(1, 9), ideal_times, marker='x', label='Идеальное время', linestyle='--')
    plt.xlabel('Количество процессов')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Время выполнения vs. количество процессов')
    plt.legend()
    plt.grid()

    # График средних вкладов
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 9), avg_rho_percent, marker='o', label='Средний вклад (ρ(n))')
    plt.xlabel('Количество процессов')
    plt.ylabel('Средний вклад (ρ(n))')
    plt.title('Средний вклад vs. количество процессов')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

