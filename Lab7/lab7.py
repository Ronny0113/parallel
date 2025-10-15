import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import time
from sklearn.datasets import make_blobs
from itertools import combinations


def plot_time_dependence(nums_processes, times):
    plt.title('Зависимость времени вычислений\nот количества выделенных процессов')
    plt.plot(nums_processes, times, label='Изменение времени по результатам эксперимента')

    # график при начальном времени (1 процесс) и постепенном разбиении на подпроцессы (эталон: гипербола)
    plt.plot(nums_processes, [times[0] / i for i in nums_processes], label='Эталонное изменение времени')
    plt.ylabel('Время выполнения в секундах')
    plt.xlabel('Количество выделенных процессов')
    plt.legend()
    plt.show()


# отображение точек по кластерам
def plot_clusters(points, clusters, title):
    plt.scatter(points[:, 0], points[:, 1], c=clusters)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()

def plot_clusters_origin(points, clusters, title):
    plt.scatter(points[:, 0], points[:, 1], c='blue')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


# матрица оценок кучностей вокруг точек с целочисленными координатами (чем больше кучность около точки, тем выше вес в соответствующей ей координате)
def get_points_weights(points, radius, min_vals, max_vals):
    points_weights = np.zeros(shape=max_vals - min_vals + 1)

    # перебор целочисленных координат
    for x in range(points_weights.shape[0]):
        for y in range(points_weights.shape[1]):
            # перебор точек, для которых ищется центр с целочисленными координатами в некоторой окрестности
            for point in points:
                # расстояние от потенциального центра до каждой из точек
                distance = np.linalg.norm(np.array((x, y)) + min_vals - point)
                if distance <= radius:
                    # увеличение веса точки отностительно других потенциальных центров
                    points_weights[x, y] += 1 / (distance + 1)

    return points_weights


# получение точки с целочисленными координатами, в окрестности которой сконцентрировано наибольшее число точек из передаваемого массива
def get_center(points, points_weights, min_vals):
    # если передается одна точка
    if points.shape[0] == 1:
        return points.squeeze()

    # получение координат точки с целочисленными координатами с максимальным весом
    center_coords = np.array(np.unravel_index(np.argmax(points_weights), points_weights.shape)) + min_vals

    return center_coords


def delete_clustered_points(points, center_coords, radius):
    points = np.copy(points)

    # Вычислим расстояние от каждой точки до выбранного центра
    points_distances = np.linalg.norm(points - center_coords, axis=1)

    # индексы точек, которые не попадают в окрестность центра
    not_clustered_points_idxs = np.where(points_distances > radius)

    # удаление кластеризованных точек
    points = points[not_clustered_points_idxs]

    return points


def get_clusters_coord_sums(points, points_clusters, centers):
    assert points.shape[0] == points_clusters.shape[0]

    clusters_coord_sums = np.zeros_like(centers)

    for cluster_idx in np.unique(points_clusters):
        points_idxs = np.where(points_clusters == cluster_idx)
        clusters_coord_sums[cluster_idx] = np.sum(points[points_idxs], axis=0)

    return clusters_coord_sums


def average_centers(points_clusters, centers, clusters_coord_sums):
    new_centers = np.copy(centers)

    for cluster_idx in np.unique(points_clusters):
        new_centers[cluster_idx] = clusters_coord_sums[cluster_idx] / np.count_nonzero(points_clusters == cluster_idx)

    return new_centers


def get_points_clusters(points, centers):
    points_clusters = np.empty(shape=points.shape[0], dtype=np.uint)

    for i in range(points.shape[0]):
        distances = np.linalg.norm(centers - points[i], axis=1)
        points_clusters[i] = np.argmin(distances)

    return points_clusters


# ДОПИСАЛ ОБЪЕДИНЕНИЕ ОЧЕНЬ БЛИЗКИХ КЛАСТЕРОВ
def merge_close_clusters(points_clusters, centers, cluster_std=1):
    assert len(np.unique(centers, axis=0)) == len(centers)

    new_points_clusters = np.copy(points_clusters)
    new_centers = np.copy(centers)

    combs = np.array(list(combinations(new_centers, 2)))

    distances = np.linalg.norm(combs[:, 0] - combs[:, 1], axis=1)

    while np.any(distances < 2.5 * cluster_std):
        min_dist_idx = np.argmin(distances)

        center1, center2 = combs[min_dist_idx]

        center1_idx = np.where((new_centers == center1).all(axis=1))[0][0]
        center2_idx = np.where((new_centers == center2).all(axis=1))[0][0]

        new_centers[center1_idx] = (center1 + center2) / 2

        new_centers = np.delete(new_centers, center2_idx, axis=0)

        new_points_clusters[np.where(new_points_clusters == center2_idx)] = center1_idx
        new_points_clusters[np.where(new_points_clusters > center2_idx)] -= 1

        combs = np.array(list(combinations(new_centers, 2)))

        distances = np.linalg.norm(combs[:, 0] - combs[:, 1], axis=1)

    return new_points_clusters, new_centers


def get_slices_borders(size, n):
    slice_size = size // n
    remainder = size % n
    slices_borders = [i * slice_size + min(i, remainder) for i in range(n + 1)]
    return slices_borders


def async_clustering(points, cluster_std, num_proc):
    # получение минимальных и максимальных целочисленных координат для перебора
    min_vals = np.int8(np.floor(np.min(points, axis=0)))
    max_vals = np.int8(np.ceil(np.max(points, axis=0)))

    # радиус окрестности вокруг потенциального центра (доля от максимального расстояния между точками, которые перебираются для выбора центра)
    radius = 0.1 * np.linalg.norm(max_vals - min_vals)

    not_clustered_points = np.copy(points)

    centers = []

    slice_bords = get_slices_borders(points.shape[0], num_proc)

    with mp.Pool(processes=num_proc) as pool:
        while np.any(not_clustered_points):
            potential_centers_weights_args = [
                (not_clustered_points[slice_bords[i]: slice_bords[i + 1]], radius, min_vals, max_vals) for i in range(num_proc)]

            potential_centers_weights = np.sum(
                np.array(pool.starmap_async(get_points_weights, potential_centers_weights_args).get()), axis=0)

            center_coords = get_center(not_clustered_points, potential_centers_weights, min_vals)

            centers.append(center_coords)

            delete_clustered_points_args = [(not_clustered_points[slice_bords[i]: slice_bords[i + 1]], center_coords, radius) for i in range(num_proc)]

            not_clustered_points = np.concatenate(pool.starmap_async(delete_clustered_points, delete_clustered_points_args).get())

        centers = np.array(centers)
        old_centers = np.copy(centers)

        get_points_clusters_args = [(points[slice_bords[i]: slice_bords[i + 1]], old_centers) for i in range(num_proc)]
        points_clusters = np.concatenate(pool.starmap_async(get_points_clusters, get_points_clusters_args).get())

        get_clusters_sums_args = [(points[slice_bords[i]: slice_bords[i + 1]], points_clusters[slice_bords[i]: slice_bords[i + 1]], centers) for i in range(num_proc)]
        clusters_coord_sums = np.sum(pool.starmap_async(get_clusters_coord_sums, get_clusters_sums_args).get(), axis=0)

        centers = average_centers(points_clusters, old_centers, clusters_coord_sums)

        while not np.array_equal(old_centers, centers):
            old_centers = np.copy(centers)

            get_clusters_sums_args = [(points[slice_bords[i]: slice_bords[i + 1]], points_clusters[slice_bords[i]: slice_bords[i + 1]], centers) for i in range(num_proc)]
            clusters_coord_sums = np.sum(pool.starmap_async(get_clusters_coord_sums, get_clusters_sums_args).get(), axis=0)

            centers = average_centers(points_clusters, centers, clusters_coord_sums)

            get_points_clusters_args = [(points[slice_bords[i]: slice_bords[i + 1]], centers) for i in range(num_proc)]
            points_clusters = np.concatenate(pool.starmap_async(get_points_clusters, get_points_clusters_args).get())

            points_clusters, centers = merge_close_clusters(points_clusters, centers, cluster_std)

    return points_clusters


def main():
    # список разного числа используемых процессов (зависит от числа доступных ЦПУ)
    nums_processes = [i + 1 for i in range(mp.cpu_count())]

    # список измерений времени для разного количества используемых процессов
    times = []

    # случайное количество генерируемых кластеров
    num_clusters = 7

    # общее количество генерируемых точек
    num_points = 3000

    cluster_std = 1.2

    # создание кластеров
    points, true_clusters = make_blobs(n_samples=num_points, centers=num_clusters, cluster_std=cluster_std, random_state=1)

    # эксперимент для разного числа процессов
    for num_proc in nums_processes:
        start_time = time.time()
        points_clusters = async_clustering(points, cluster_std, num_proc)
        end_time = time.time()
        times.append(end_time - start_time)

    plot_clusters_origin(points, true_clusters, title=f'Исходный график')
    plot_clusters(points, points_clusters, title=f'Текущий график с числом кластеров: {len(np.unique(points_clusters))}')
    plot_time_dependence(nums_processes, times)


if __name__ == '__main__':
    main()


