import numpy as np

a = np.array([[1,2,3],[2,3,4]])

z = np.zeros((2,1), dtype=int)

t = np.append(a, z, axis=1)
