import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Определяем функцию
def f(x, y):
    return np.sin(x + y) * np.cos(y) - np.cos(x + 2 * y)

# Создаем сетку для x и y
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# Вычисляем значения функции
z = f(x, y)

# Создаем 3D график
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# Добавляем подписи и заголовок
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('f(x, y)')
ax.set_title('График функции f(x, y) = sin(x+y) * cos(y) - cos(x + 2y)')

# Показываем график
plt.show()
