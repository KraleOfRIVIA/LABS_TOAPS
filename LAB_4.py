import numpy as np
import matplotlib.pyplot as plt
import Lab3

fig, ax = plt.subplots()
x = [-8, -6, -3.5, -3, -2.5, 0, 2, 2.5, 4, 6.5]
y = [-1, 3, 6.5, 4, 2, 4, 4.5, 1, -2, 1]
x1 = [-10, -9, -5, -1, 1.5, 3, 5, 9]

X = [-9.5, -6.5, -4, -2.5, -0.5, 1.5, 3, 4.5, 9.5]
Y = [5.5, 1, -4.5, -2, -5, -2.5, 0, 1.5, -1.5]
X1 = [-8, -5, -3, -1.5, 0.5, 4, 8]

A1 = np.column_stack([X, Y])
cord_x_and_y = np.column_stack([x, y])
sort = np.argsort(cord_x_and_y[:, 0])
cord_x_and_y = cord_x_and_y[sort]
X1.sort()
[x, y] = np.hsplit(cord_x_and_y, 2)
ax.plot(cord_x_and_y)
print(cord_x_and_y)
# print("\n")
# plt.show()
N = 10


def interval(x, x1):
    n = np.size(x)
    n1 = np.size(x1)
    k = 1
    itr = np.zeros((n1, 1))
    for i in n1:
        if (x1[i] < x[1]):
            itr[i] = 0
            continue
        if (x1[i] > x[n]):
            itr[i] = n
        j = 1
        while (j <= n - 1):
            if (x1[i] >= x[j] and x1[i] <= (j + 1)):
                itr[i] = j
                break
            else:
                j += 1
    return itr


def square_val(x, y, x1, itr):
    global cord_x_and_y
    n = np.size(x)
    n1 = np.size(x1)
    a = np.zeros(n1)
    b = np.zeros(n1)
    c = np.zeros(n1)
    y1 = np.zeros((1, n1))
    i = 1
    while (i <= n - 2):
        x_new = np.arange([x[i] ^ 2, x[i], 1], [x[i + 1] ^ 2, x[i + 1], 1], [x[i + 2] ^ 2, x[i + 2], 1])
        y_new = np.arange([y[i]], [y[i + 1]], [y[i + 2]])
        A = np.column_stack([x_new, y_new])
        C = Lab3.GaussStraightFunc(A)
        a[i] = C[1]
        b[i] = C[2]
        c = C[3]
        i += 1
    a[n - 1] = a[n - 2]
    b[n - 1] = b[n - 2]
    c[n - 1] = c[n - 2]
    i = 1
    while (i <= n1):
        j = itr[i]
        if ((j == 0) or (j == n)):
            y1[i] = np.NAN
        if ((j>0) and (j<n)):
            y1[i] = a[j]*x1[i]^2 + b[j]*x1[i]+c[j]
        i+=1
    return y1
N = 10
n = np.size((x,2))
x2 = np.zeros((1,1))
x2[1] = x[1]
for i in range(1,n-2):
    h = (x[i+1]-x[i])/N
#     доделать . спросить
