import numpy as np
import matplotlib.pyplot as plt

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
    ax.set_title('График функции p(theta, phi) = cos(2phi)')
    plt.show()

if __name__ == "__main__":
    plot_spherical_function()
