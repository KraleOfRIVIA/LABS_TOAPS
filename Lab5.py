import numpy as np
import matplotlib.pyplot as plt
from math import nan
import itertools

x = [-8, - 6, - 3.5, - 3, - 2.5, 0, 2, 2.5, 4, 6.5]
y = [-1, 3, 6.5, 4, 2, 4, 4.5, 1, - 2, 1]
x1 = [-10, - 9, - 5, - 1, 1.5, 3, 5, 9]
# x = [-9.5, -6.5, -4, -2.5, -0.5, 1.5, 3, 4.5, 9.5]
# y = [5.5, 1, -4.5, -2, -5, -2.5, 0, 1.5, -1.5]
# x1 = [-8, -5, -3, -1.5, 0.5, 4, 8]
cord_x_and_y = np.column_stack([x, y])
sort = np.argsort(cord_x_and_y[:, 0])
cord_x_and_y = cord_x_and_y[sort]
x1.sort()
[x, y] = np.hsplit(cord_x_and_y, 2)
x = np.hstack(x)
y = np.hstack(y)
n = 5
h[1:n] = x[2:n+1] - x[1:n]
A[1] = 0
A[2:n-1] = h[2:n-1]
B[1:n-1] = 2*(h[1:n-1]+h[2:n])
C[1:n-2] = h[2:n-1]
C[n-1] = 0
D = np.zeros((np.arange(1.,(n-1))))
def progom(x, y):
    n = np.size(x) - 1
    h[1:n] = x[2:n+1] - x[1:n]
    A[1] = 0
    A[2:n-1] = h[2:n-1]
    B[1:n-1] = 2*(h[1:n-1]+h[2:n])
    C[1:n-2] = h[2:n-1]
    C[n-1] = 0
    D = np.zeros((np.arange(1.,(n-1))))
    print("хуй")
def interval(x, x1):
    n = np.size(x)
    n1 = np.size(x1)
    itr = np.zeros(n1)
    j = 0
    for i in range(0, n1 - 1):
        if (x1[i] < x[0]):
            itr[i] = 0
            continue
        if (x1[i] > x[n - 1]):
            itr[i] = n - 1
        while (j <= n - 2):
            if (x1[i] >= x[j]) and (x1[i] <= x[j + 1]):
                itr[i] = j
                i += 1
                break
            else:
                j += 1

    return itr
