import time
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
# генерирование бинарного дерева с ветвями 1 и -1 и случайными коэффициентами
def get_bin_graph_with_random_coefs(alpha_range, beta_range, N):
# формирование таблицы всех конфигураций, состоящей из 1 и -1 (каждая строка - набор конфигураций А1, А2, ..., АN)
    values = np.array([-1, 1])
    all_confs = np.array(np.meshgrid(*[values] * N)).T.reshape(-1, N)

    # массив из N коэффициентов при конфигурациях
    alpha_arr = np.random.randint(alpha_range[0], alpha_range[1] + 1, size=N)

    # треугольная матрица из (N^2 - N)/2 коэффициентов при попарных перемножениях конфигураций (значения на главной диагонали и ниже ее равны 0)
    beta_matrix = np.triu(np.random.randint(beta_range[0], beta_range[1], size=(N, N)), k=1)

    return all_confs, alpha_arr, beta_matrix


# получение попарных перемножений элементов конфигураций
def get_pairwise_conf_items_mult(conf):
    ai, aj = np.meshgrid(conf, conf)
    A_pairwise_mult = ai * aj
    return A_pairwise_mult


# получение E
def get_bin_graph_result(conf, alpha_arr, beta_matrix):
    A_pairwise_mult = get_pairwise_conf_items_mult(conf)
    return np.sum(alpha_arr * conf) + np.sum(beta_matrix * A_pairwise_mult)


# подбор конфигурации, при которой выходное значение (E) - минимально
def get_confs_to_minimize(confs, alpha_arr, beta_matrix):
    # список результатов вычислений для каждого набора конфигураций
    E_arr = []

    for A in confs:
        E = get_bin_graph_result(A, alpha_arr, beta_matrix)
        E_arr.append(E)

    # искомое минимальное значение
    min_E = np.min(E_arr)

    # индексы всех минимальных значений
    indices = np.where(E_arr == min_E)
    best_confs = confs[indices]

    return min_E, best_confs, E_arr


# список разного числа используемых процессов (зависит от числа доступных ЦПУ)
nums_processes = [i + 1 for i in range(cpu_count())]

# список измерений времени для разного количества используемых процессов
times = []

# список минимальных значений Е для каждого из процессов
min_E_list = []

# список лучших найденных конфигураций для каждого из процессов
best_confs_list = []

# число уровней
N = 22

# пределы генерируемых значений альфа и бета
alpha_range = (-10, 10)
beta_range = (-10, 10)

# создание N-уровнего бинарного графа со случайными коэффициентами соответствующего диапазона
all_confs, alpha_arr, beta_matrix = get_bin_graph_with_random_coefs(alpha_range, beta_range, N)


def main():
    # эксперимент для каждого из количества процессов, включая использование основного
    for n in nums_processes:
        # создание пула дополнительных процессов
        with Pool(processes=n) as pool:
            # разбитие всего числа конфигураций на n частей
            splitted_confs = np.array_split(all_confs, n, axis=0)

            # аргументы для передачи в функцию, для которой проводится параллелизация
            args = [(confs, alpha_arr, beta_matrix) for confs in splitted_confs]

            # каждый из процессов будет генерировать равное количество точек
            start_time = time.time()

            # вычисления в рамках пула
            pool_results = pool.starmap_async(get_confs_to_minimize, args).get()

            # закрытие пула
            pool.close()

            # ожидание завершения дополнительных процессов
            pool.join()

            end_time = time.time()

            # сортировка результатов
            pool_results = sorted(pool_results, key=lambda x: x[0])

            # выбор минимального (в начале списка)
            min_E = pool_results[0][0]
            best_confs = pool_results[0][1]

            times.append(end_time - start_time)
            min_E_list.append(min_E)
            best_confs_list.append(best_confs)

    print('Сгенерированные коэффициенты альфа:\n', alpha_arr)
    print('Сгенерированные коэффициенты бета:\n', beta_matrix)

    for n in nums_processes:
        print(
            f'CPU: {n}, t: {times[n - 1]:.2f}, min: {min_E_list[n - 1]} при {' или '.join(map(str, best_confs_list[n - 1]))}')

    plt.title('Зависимость времени вычислений\nот количества выделенных процессов')
    plt.plot(nums_processes, times, label='Изменение времени по результатам эксперимента')

    # график при начальном времени (1 процесс) и постепенном разбиении на подпроцессы (эталон: гипербола)
    plt.plot(nums_processes, [times[0] / i for i in nums_processes], label='Эталонное изменение времени')
    plt.ylabel('Время выполнения в секундах')
    plt.xlabel('Количество выделенных процессов')
    plt.legend()
    plt.show()

    min_E, best_confs, E_arr = get_confs_to_minimize(all_confs, alpha_arr, beta_matrix)

    # меры хаоса
    T = np.arange(1, 1000)

    # максимальная бета по модулю
    max_index = np.argmax(np.abs(beta_matrix))
    max_coords = np.unravel_index(max_index, beta_matrix.shape)
    i_abs_beta_max, j_abs_beta_max = map(int, max_coords)

    # зависимость средневзвешенного <Ai, Aj> от меры хаоса
    AiAj_func = []

    # функция меры хаоса
    T_func = []

    for t in T:
        # статистический вклад
        p = np.exp(-(E_arr - min_E) / t)

        # сумма статистических вкладов
        z = np.sum(p)

        # нормализация
        p_norm = p / z

        AiAj_sum = 0
        for i in range(2 ** N):
            A_pairwise_mult = get_pairwise_conf_items_mult(all_confs[i])
            AiAj_sum += A_pairwise_mult[i_abs_beta_max, j_abs_beta_max] * p_norm[i]

        AiAj_func.append(AiAj_sum)

        T_func.append(np.sum(E_arr * p_norm))

    plt.title('Функция меры хаоса')
    plt.plot(T, T_func)
    plt.xlabel('T')
    plt.ylabel('<E>(T)')
    plt.show()

    plt.title('Зависимость средневзвешенного <Ai, Aj> от меры хаоса')
    plt.plot(T, AiAj_func)
    plt.xlabel('T')
    plt.ylabel('<Ai, Aj>')
    plt.show()


if __name__ == '__main__':
    main()
