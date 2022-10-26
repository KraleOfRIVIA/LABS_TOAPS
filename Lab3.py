import numpy as np
import matplotlib.pyplot as plt
A = np.array([[0.51, -1.9, -0.3], [1.9, 3.2, -0.5], [3.6, 0.3, -1.4]])
B = np.array([1.1, 1.2, 5.2])
X = np.linalg.inv(A).dot(B)
# print(X)
B = A * X
# print(B)
# eps = np.sum(np.abs(B - A * X))
# print(eps)
A = np.array([[0.51, -1.9, -0.3, 1.1], [1.9, 3.2, -0.5, 1.2], [3.6, 0.3, -1.4, 5.2]])
B = np.array([1.1, 1.2, 5.2])


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



# GaussStraightFunc(A)
# print(A)
A = np.array([[3.6, -1.9, -1.4], [1.9, 3.2, -0.5], [0.51, 0.3, -0.3]],dtype=float)
B = np.array([1.1, 1.2, 5.2],dtype=float)


def itera1():
    global A, B
    mat = A
    nextVector = B
    for nrow in range(len(mat)):
        nextVector[nrow] /= (mat[nrow, nrow])
        mat[nrow, :] /= -mat[nrow, nrow]
        mat[nrow, nrow] = 0
    A = mat
    B = nextVector


# print(itera1(A,B))
eps = 1e-3
k = 1
itera1()
X = B
X1 = np.dot(A, X) + B
kmax = 100
# A1 = np.column_stack([A, B])
D = ((np.max(np.abs(X1 - X))) > eps)
while (D > eps) and (k < kmax):
    X = X1
    X1 = np.dot(A, X) + B
    k = k + 1
    D = ((np.max(np.abs(X1 - X))) > eps)
    # print("Root:")
    # print(X)
    # print("Number of iterations:")
    # print(k)
print(X1)
print(k)