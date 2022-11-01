import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x = [-8, -6, -3.5, -3, -2.5, 0, 2, 2.5, 4, 6.5]
y = [-1, 3, 6.5, 4, 2, 4, 4.5, 1, -2, 1]
x1 = [-10, -9, -5, -1, 1.5, 3, 5, 9]

X = [-9.5, -6.5,-4, -2.5, -0.5, 1.5, 3, 4.5, 9.5]
Y = [5.5, 1, -4.5, -2, -5, -2.5, 0, 1.5, -1.5]
X1 = [-8, -5, -3, -1.5, 0.5, 4, 8]

A1 = np.column_stack([X, Y])
cord_x_and_y = np.column_stack([x, y])
sort = np.argsort(cord_x_and_y[:, 0])
cord_x_and_y = cord_x_and_y[sort]
ax.plot(cord_x_and_y)
print(cord_x_and_y)
print("\n")
plt.show()
