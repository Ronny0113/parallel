import matplotlib.pyplot as plt

# Данные
threads = list(range(1, 9))  # Количество потоков от 1 до 8
times = [513, 290, 197, 169, 175, 182, 195, 212]  # Время выполнения в секундах

# Вычисление значений для кривой
base_time = times[0]  # Первый элемент времени
theoretical_times = [base_time / t for t in threads]

# Создание графика
plt.figure(figsize=(10, 6))
plt.plot(threads, times, label='Время выполнения')
plt.plot(threads, theoretical_times, label='Идеальное время')

# Настройка графика
plt.title('Зависимость времени вычислений\nот количества выделенных процессов')
plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения (сек)')
plt.xticks(threads)  # Установка меток по оси X
plt.grid(True)
plt.legend()

# Показать график
plt.show()
