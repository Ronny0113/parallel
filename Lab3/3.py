import math
import multiprocessing
import itertools

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


'''В голосовании участвуют n=24 агентов, каждый из которых может проголосовать за
 (-1) либо проголосовать против (1). Существует 2^n возможных конфигураций (событий).
  Кроме этого, то, как проголосует агент зависит от голосов остальных агентов.
   Также существует мера хаоса T, которая влияет на результат.'''

def flatten_list(list_2d):# сюда попадут значения со всех потоков
    flat_list = []

    for element in list_2d:#достаём из листов элементы и бросаем в общий, чтобы не было многоэтажности
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def get_combination(index, size):
    return [-1 if x == '0' else 1 for x in f'{{0:0{size}b}}'.format(index)]


def calc_tension(combination_range, actor_factor_list, actor_parir_factor_list):
    tension_list = []

    size = len(actor_factor_list)

    for combination_id in range(combination_range[0], combination_range[1]):
        combination = get_combination(combination_id, size)

        actor_sum = sum(
            [i*j for i, j in zip(combination, actor_factor_list)])
        pair_sum = sum([actor_parir_factor_list[i] * pair[0] * pair[1]
                        for (i, pair) in enumerate(itertools.combinations(combination, 2))])
        tension = actor_sum + pair_sum

        tension_list.append(tension)

    return tension_list


def calc_average(chaos_list, tension_list, delta_tension): # СЧИТАЕМ СРЕДНЕЕ (с учётом хаоса)
    average_tension_list = []

    for chaos in chaos_list:
        average_tension = 0
        rho_sum = 0

        for index in range(len(delta_tension)):
            rho = math.exp(delta_tension[index] / -chaos)
            rho_sum += rho
            average_tension += tension_list[index] * rho

        average_tension_list.append(average_tension / rho_sum)

    return average_tension_list


def main(actor_count, process_count):
    combination_count = 2 ** actor_count
    combinations_per_process = combination_count // process_count

    actor_factor_list = np.random.uniform(-10, 10, actor_count)
    actor_pair_factor_list = np.random.uniform(
        -10, 10, int((actor_count ** 2 - actor_count) / 2))

    combination_range_list = [(i * combinations_per_process, (i + 1)
                               * combinations_per_process) for i in range(process_count)]

    process_data = [(combination_range, actor_factor_list,
                     actor_pair_factor_list) for combination_range in combination_range_list]
    #ПЕРВЫЙ ПАРАЛЕЛИЗМ
    def parrallel_tension_calculation():
        with multiprocessing.Pool(process_count) as pool:
            return flatten_list(pool.starmap(
                calc_tension, process_data))

    tension_list = parrallel_tension_calculation()

    min_tension = min(tension_list) #Это тот самый E0
    average_tension = sum(tension_list) / len(tension_list) #средняя

    chaos = np.linspace(0.001, 1000, 1500)# точки в заданном диапазоне
    chaos_process = np.array_split(chaos, process_count)# слеиваем
    delta_tension = [tension - min_tension for tension in tension_list]

    chaos_process_data = [(chaos_process[i], tension_list, delta_tension)
                          for i in range(process_count)]
    #ВТОРОЙ ПАРАЛЕЛИЗМ
    def parrallel_average_calculation():
        with multiprocessing.Pool(process_count) as pool:
            return flatten_list(pool.starmap(
                calc_average, chaos_process_data))

    average_list = parrallel_average_calculation()

    return {
        'result': (chaos, average_list),
        'min': min_tension,
        'average': average_tension,
    }


if __name__ == '__main__':
    result = main(10, 4)
    print(result)
