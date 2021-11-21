import numpy as np

a = np.array([[[1,1,1,9],[2,2,2,9]],[[1,1,1,1],[2,2,2,2]]])
print(a)
print(a.shape)

print(np.delete(a, 3, 2))
