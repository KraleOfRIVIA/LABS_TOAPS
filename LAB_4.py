import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x = np.array([-8, -6, -3.5, -3, -2.5, 0, 2, 2.5, 4, 6.5])
y = np.array([-1, 3, 6.5, 4, 2, 4, 4.5, 1, -2, 1])
x1 = np.array([-10, -9, -5, -1, 1.5, 3, 5, 9])
X = [-9.5, -6.5, -4, -2.5, -0.5, 1.5, 3, 4.5, 9.5]
Y = [5.5, 1, -4.5, -2, -5, -2.5, 0, 1.5, -1.5]
X1 = [-8, -5, -3, -1.5, 0.5, 4, 8]
k = np.interp(x1, x, y, period=360)
# print(k)
A1 = np.column_stack([X, Y])
cord_x_and_y = np.column_stack([x, y])
sort = np.argsort(cord_x_and_y[:, 0])
cord_x_and_y = cord_x_and_y[sort]
X1.sort()
[x, y] = np.hsplit(cord_x_and_y, 2)
x = np.hstack(x)
y = np.hstack(y)
# ax.plot(x1, k)
# print(cord_x_and_y)
# print("\n")
# plt.show()
N = 10


def interval(x, x1):
    n = np.size(x) - 1
    n1 = np.size(x1)
    irt = np.zeros((n1))
    for i in range(0, n1):
        if (x1[i] < x[0]):
            irt[i] = 0
            continue
        if (x1[i] > x[n]):
            irt[i] = n
        j = 1
        while (j <= n - 1):
            if(x1[i]<x[i]):
                j+=1
            else:
                irt[i] = j
                break

    return irt


def GaussStraightFunc(matrix):
    for nrow in range(len(matrix)):
        # pivot равен номеру столбца
        # nrow равен номеру строки
        # np.argmax возвращает номер строки с максимальным элементом в уменьшенной матрице
        # которая начинается со строки nrow. Поэтому нужно прибавить nrow к результату
        pivot = nrow + np.argmax(abs(matrix[nrow:, nrow]))
        if pivot != nrow:
            # swap
            matrix[[nrow, pivot]] = matrix[[pivot, nrow]]
        row = matrix[nrow]
        divider = row[nrow]  # диагональный элемент
        if abs(divider) < 1e-10:
            # почти ноль на диагонали
            raise ValueError("Решений нет")
        # делим на диагональный элемент.
        row /= divider
        # теперь надо вычесть приведённую строку из всех нижележащих строчек
        for lower_row in matrix[nrow + 1:]:
            factor = lower_row[nrow]  # элемент строки в колонке nrow
            lower_row -= factor * row  # вычитаем, чтобы получить ноль в колонке nrow
    # приводим к диагональному виду
    reverse(matrix)
    return matrix


def reverse(matrix):
    # перебор строк в обратном порядке
    for nrow in range(len(matrix) - 1, 0, -1):
        row = matrix[nrow]
        for upper_row in matrix[:nrow]:
            factor = upper_row[nrow]
            upper_row -= factor * row
    return matrix


def square_val(x, y, x1, irt):
    global cord_x_and_y
    n = np.size(x)
    n1 = np. size(x1)
    a = np.zeros(n1)
    b = np.zeros(n1)
    c = np.zeros(n1)
    y1 = np.zeros(n1)
    # print(y1[3])
    i = 0
    while (i <= (n - 3)):
        x_new = np.array([[x[i] ** 2, x[i], 1], [x[i + 1] ** 2, x[i + 1], 1], [x[i + 2] ** 2, x[i + 2], 1]],dtype=float)
        y_new = np.array([y[i], y[i + 1], y[i + 2]])
        M = np.column_stack([x_new, y_new])
        C = GaussStraightFunc(M)
        # print(C[1])
        a[i] = C[0][3]
        b[i] = C[1][3]
        c[i] = C[2][3]
        i += 1
    a[n - 3] = a[n - 4]
    b[n - 3] = b[n - 4]
    c[n - 3] = c[n - 4]
    i = 0
    for i in range(0,n1-2):
        j = irt[i]
        if ((j == -1) or (j == n)):
            y1[i] = -1
        if ((j >= 0) and (j < n)):
            y1[i] = (a[j] * (x1[i]) ** 2 + b[j] * x1[i] + c[j])
    return y1


irt = interval(x, x1)
irt = irt.astype(np.int64)
irt = np.hstack(irt)
y1 = square_val(x, y, x1, irt)
# print(x1)
# print("\n")
# print(y1)
a = min(x)
b = max(x)
x2 = np.arange(a,b,1)
itr = interval(x,x2)
itr = itr.astype(np.int64)
itr = np.hstack(itr)
y2 = square_val(x,y,x2,itr)
ax.plot(x1,y1)
plt.grid()
plt.show()

    # доделать . спросить
