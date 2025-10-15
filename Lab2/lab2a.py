import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing

R = 1.0  # радиус полусферы
N = 1000000  # количество случайных точек


def analytical_volume():
    return (2 / 3) * np.pi * R ** 3


def monte_carlo_volume(num_points):
    count_inside = 0
    for _ in range(num_points):
        x, y, z = np.random.uniform(-R, R, 3)
        if x ** 2 + y ** 2 + z ** 2 <= R ** 2 and z >= 0:  # точка внутри полусферы
            count_inside += 1
    volume_cube = (2 * R) ** 3
    return volume_cube * (count_inside / num_points)


def worker(num_points):
    return monte_carlo_volume(num_points)


def plot_3d_sphere():
    # Создаем сетку сферических координат
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))

    # Создаем 3D-график
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, alpha=0.5, rstride=5, cstride=5, color='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Полусфера радиуса R')
    plt.show()


if __name__ == "__main__":
    num_processors = range(1, 9)
    times = []

    # Аналитический объем
    analytic_vol = analytical_volume()
    print(f"Аналитический объем: {analytic_vol}")

    # Визуализация фигуры
    plot_3d_sphere()

    for num in num_processors:
        start_time = time.time()

        with multiprocessing.Pool(processes=num) as pool:
            results = pool.map(worker, [N // num] * num)  # делим работу

        estimated_volume = sum(results) / num  # усредняем объемы
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(
            f"Процессоров: {num}, Время выполнения: {elapsed_time:.4f} сек, Приблизительный объем: {estimated_volume}")

    # Построение графика
    ideal_times = [times[0] / i for i in num_processors]  # идеальное время
    plt.plot(num_processors, times, marker='o', label='Фактическое время')
    plt.plot(num_processors, ideal_times, marker='x', linestyle='--', label='Идеальное время (t/n)')

    plt.title('Зависимость времени выполнения от количества процессоров')
    plt.xlabel('Количество процессоров')
    plt.ylabel('Время выполнения (сек)')
    plt.xticks(num_processors)
    plt.legend()
    plt.grid()
    plt.show()
