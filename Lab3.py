import numpy as np
import matplotlib.pyplot as plt

# from numpy.linalg import inv
# from numpy.linalg import dot
A = np.array([[0.51, -1.9, -0.3], [1.9, 3.2, -0.5], [3.6, 0.3, -1.4]])
B = np.array([1.1, 1.2, 5.2])
X = np.linalg.inv(A).dot(B)
# print(X)
B = A * X
# print(B)
eps = np.sum(np.abs(B - A * X))
print(eps)
A = np.array([[0.51, -1.9, -0.3, 1.1], [1.9, 3.2, -0.5, 1.2], [3.6, 0.3, -1.4,5.2]])
B = np.array([1.1, 1.2, 5.2])


def gaussPivotFunc(matrix):
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
        divider = row[nrow] # диагональный элемент
        if abs(divider) < 1e-10:
            # почти нуль на диагонали. Продолжать не имеет смысла, результат счёта неустойчив
            raise ValueError(f"Матрица несовместна. Максимальный элемент в столбце {nrow}: {divider:.3g}")
        # делим на диагональный элемент.
        row /= divider
        # теперь надо вычесть приведённую строку из всех нижележащих строчек
        for lower_row in matrix[nrow+1:]:
            factor = lower_row[nrow] # элемент строки в колонке nrow
            lower_row -= factor*row # вычитаем, чтобы получить ноль в колонке nrow
    # приводим к диагональному виду
    make_identity(matrix)
    return matrix


def make_identity(matrix):
    # перебор строк в обратном порядке
    for nrow in range(len(matrix) - 1, 0, -1):
        row = matrix[nrow]
        for upper_row in matrix[:nrow]:
            factor = upper_row[nrow]
            upper_row -= factor * row
    return matrix


# X1 = np.linalg.solve(A, B)
# print(X1)
gaussPivotFunc(A)
print(A)