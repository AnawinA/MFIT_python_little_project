import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2)
b = np.array([[1, 2, 3], [4, 5, 6]]).reshape(2, 3)
print(a, b, sep="\n\n")
print("---")

print(np.dot(a, b))
print("- same to -")
print(a @ b)
