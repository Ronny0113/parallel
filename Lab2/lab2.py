import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


def monte_carlo_volume(n_points):
    count_inside = 0

    # Параметры параллелепипеда
    x_bound = [-1, 1]
    y_bound = [-1, 1]
    z_bound = [-1, 1]

    for _ in range(n_points):
        # Генерация случайных точек
        x = np.random.uniform(*x_bound)
        y = np.random.uniform(*y_bound)
        z = np.random.uniform(*z_bound)

        # Вычисляем полярные координаты
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if r == 0:
            continue

        phi = np.arccos(z / r)  # Угол от положительной оси Z
        # Параметрическая функция радиуса
        surf_point_r = abs(np.cos(2 * phi))

        if r <= surf_point_r:
            count_inside += 1

    volume = (count_inside / n_points) * 8
    return volume


def calculate_volumes(processors, n_points):
    # Вычисление объема методом Монте-Карло с использованием нескольких процессов
    with Pool(processors) as p:
        # Деля текущее количество точек между процессами
        points_per_process = n_points // processors
        volumes = p.map(monte_carlo_volume, [points_per_process] * processors)

    return np.mean(volumes), volumes  # Возвращаем также индивидуальные объемы


def time_function(processors, n_points):
    start_time = time.time()
    volume, individual_volumes = calculate_volumes(processors, n_points)
    elapsed_time = time.time() - start_time
    print(f'Процессоров: {processors}, Оценка объёма: {volume:.4f}, Время выполнения: {elapsed_time:.4f} секунд')
    return elapsed_time, volume  # Возвращаем время и объем


def plot_performance():
    n_points = 1000000  # Общее количество точек для Монте-Карло
    processor_counts = range(1, 9)  # От 1 до 8 процессоров
    times = []
    ideal_times = []

    for pc in processor_counts:
        elapsed_time, volume = time_function(pc, n_points)
        times.append(elapsed_time)
        ideal_times.append(times[0] / pc)  # Идеальное время выполнения

    # Построим график зависимости времени от количества процессоров
    plt.figure()
    plt.plot(processor_counts, times, marker='o', label='Фактическое время', color='blue')
    plt.plot(processor_counts, ideal_times, marker='x', linestyle='dashed', label='Идеальное время (t/n)', color='orange')
    plt.xlabel('Количество процессоров')
    plt.ylabel('Время выполнения (с)')
    plt.title('Зависимость времени выполнения от количества процессоров')
    plt.legend()
    plt.grid()
    plt.show()


def plot_spherical_function():
    # Определяем углы
    phi = np.linspace(0, np.pi, 100)  # угол от 0 до π
    theta = np.linspace(0, 2 * np.pi, 100)  # угол от 0 до 2π
    phi, theta = np.meshgrid(phi, theta)  # создаем сетку углов

    # Значение функции в сферических координатах
    r = np.cos(2 * phi)  # p(theta, phi) = cos(2phi)

    # Преобразуем сферические координаты в декартовые
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Построение графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7, rstride=5, cstride=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('График функции r(theta, phi) = cos(2phi)')
    plt.show()


if __name__ == "__main__":
    # Построение графика функции
    plot_spherical_function()

    # Построение графика зависимости времени выполнения от количества процессоров
    plot_performance()
