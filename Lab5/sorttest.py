import numpy as np

a = np.array([5, 6, 7])
b = np.array([4, 5])
result = np.concatenate((a, b))
print(result)  # Output: [1 2 3 4 5]