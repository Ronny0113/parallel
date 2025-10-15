import numpy as np
from scipy.integrate import dblquad

# Функция для r(theta, phi)
def r(phi):
    return np.cos(2 * phi)

# Интегрируем по φ и θ
def integrand(phi, theta):
    return r(phi)**2 * np.sin(phi)

# Пределы интегрирования
theta_lower = 0
theta_upper = 2 * np.pi
phi_lower = 0
phi_upper = np.pi / 2

# Выполняем двойной интеграл
volume, error = dblquad(integrand, theta_lower, theta_upper, lambda _: phi_lower, lambda _: phi_upper)

# Умножаем на 2π для final объёма
# (поскольку интегрируем по θ от 0 до 2π, тогда это уже учтено в интеграле)
print("Объём фигуры:", volume)
