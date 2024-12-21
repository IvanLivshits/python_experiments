import numpy as np
import matplotlib.pyplot as plt

# Crate the list of some numbers
lst = [0, 1, 2, 3, 4, 5]
print(type(lst))

# Convert the list to the numpy array
arr = np.array(lst)
print(type(arr))

# Differnt ways to create a numpy array
arr1 = np.arange(1, 10, 1); # start, stop, step
print(arr1)

# Reshape this array to 3x3
arr2 = arr1.reshape(3, 3)
print(arr2)

# Conveert this into float32 data type
arr3 = arr2.astype(np.float32)
print(arr3)

# Create a plt visualisation
x = np.linspace(-10, 10, 100)
y = x ** 2

plt.plot(x, y)
plt.show()

arr4 = np.array([[[1, 2, 3]],
[[4, 5, 6]],
[[7, 8, 9]]], dtype=np.float32)
print(arr4.ndim)