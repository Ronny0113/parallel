import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
import random

X_MIN = -100
X_MAX = 100
Y_MIN = -100
Y_MAX = 100
N_MIN = 1800
K = 4000
P = K // 2
R_N = X_MAX // 20
h = 10

N_ITERATIONS = 15


def f(point):
    x, y = point
    return np.sin(x + y) * np.cos(y) + np.cos(x + 2 * y)


def mutate(x):
    sign_h = 1 - np.random.randint(0, 1, size=2) * 2
    return x + h * sign_h


def cross(x):
    l = x.shape[0]
    alpha = np.random.uniform(0.1, 0.9, l)
    new_points = np.zeros((l, 2))
    for i in range(l):
        ind = np.random.choice(l, size=2, replace=False)
        x1, y1 = x[ind[0]]
        x2, y2 = x[ind[1]]
        new_points[i] = np.array([x1 * (1 - alpha[i]) + x2 * alpha[i], y1 * (1 - alpha[i]) + y2 * alpha[i]])
    return new_points


def find_min(queue, points):
    while True:
        # Случайное задание точек
        points = np.random.uniform(X_MIN, X_MAX, size=(K, 2))

        # N_ITERATIONS эволюционных шагов
        for i in range(N_ITERATIONS):
            # Эволюционный шаг
            points = np.array(sorted(points, key=lambda x: f(x)))

            points = points[:P]

            step_type = random.randint(0, 1)
            if step_type:
                points = np.concatenate((points, mutate(points)), axis=0)
            else:
                points = np.concatenate((points, cross(points)), axis=0)

        min_points = []
        points = np.array(sorted(points, key=lambda x: f(x)))

        points = points[:P]
        z = np.array([f(point) for point in points])

        # Отбор точек
        while True:
            for point in points:

                distances = np.linalg.norm(points - point, axis=1)
                suitable_points = distances < 10
                if suitable_points.sum() < 10:
                    continue
                min_points.append(point)
                points = points[~suitable_points]
                break
            else:
                break

        min_list = queue.get()
        if len(min_list) >= 1800:
            queue.put(min_list)
            break

        if len(min_points) == 0:
            queue.put(min_list)
            continue

        if len(min_list) > 0:
            min_points_arr = np.array(min_points)
            global_mins = np.array(min_list)
            distances = np.linalg.norm(min_points_arr[:, np.newaxis] - global_mins, axis=-1)
            suitable_distance_id = np.all(distances > 3, axis=1)
            suitable_points = min_points_arr[suitable_distance_id].tolist()
            queue.put(min_list + suitable_points)
            print(len(min_list))
        else:
            queue.put(min_points)


def calculate(target_function, procs_num, queue):
    # список процессов
    processes_list = []

    # запуск процессов
    for _ in range(procs_num):
        p = multiprocessing.Process(target=target_function, args=(queue, ""))
        processes_list.append(p)
        p.start()
    # Ждем, пока все процессы завершат работу.
    for p in processes_list:
        p.join()


def main():
    # Создание очереди
    m = multiprocessing.Manager()
    queue = m.Queue()

    time_list = [0] * 8
    proc_list = [i for i in range(1, 9)]
    n_exp = 1

    for _ in range(n_exp):

        for proc in proc_list:
            queue.put([])
            start = time.time()
            calculate(find_min, proc, queue)
            end = time.time()
            t = end - start
            print(f"t для {proc} проц: {t}")

            time_list[proc - 1] += t / n_exp

    mins = queue.get()
    mins = np.array(mins)
    z = np.array([f(point) for point in mins])
    print(max(z))
    print(np.mean(z))
    plt.scatter(mins[:, 0], mins[:, 1], s=0.9)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(mins[:, 0], mins[:, 1], z, color="red", s=0.9)
    ax.set_zlabel('Z')
    plt.show()

    plt.ylabel("время, с")
    plt.xlabel("количество потоков")
    plt.title("Зависисмость времени от количества потоков")
    plt.plot(range(1, 9), time_list)
    plt.plot(range(1, 9), [time_list[0] / i for i in range(1, 9)], label="1/T1")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


