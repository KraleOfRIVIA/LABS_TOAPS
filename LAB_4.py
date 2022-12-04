import numpy as np
import matplotlib.pyplot as plt
from math import nan, pi
import pandas as pan
fig, ax = plt.subplots()
x=[0, 5.5, 8.5, 10.5 ,12.5, 14, 15, 16.5 ,18]
y=[5.5 ,7.5, 9.5, 11.5, 13, 12, 11, 12, 13]
x1=[2, 7 ,11.5, 13, 17, 19, 21]
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
def interval(x, x1):
    n = np.size(x)
    n1 = np.size(x1)
    itr = np.zeros(n1,dtype=int)
    j = 0
    for i in range(0, n1 - 1):
        if (x1[i] < x[0]):
            itr[i] = -1
            continue
        if (x1[i] > x[n - 1]):
            itr[i] = - 1
        while (j <= n - 2):
            if (x1[i] >= x[j]) and (x1[i] <= x[j + 1]):
                itr[i] = j
                i += 1
                break
            else:
                j += 1
    itr[-1] = itr[-2]
    return itr

def square_val(x,y,x1,itr):
    n = np.size(x) - 1
    n1 = np.size(x1)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    y1 = np.zeros(n1)
    i = 0
    while i<=n-2:
        x_new = [[x[i]**2,x[i],1],[x[i+1]**2,x[i+1],1],[x[i+2]**2,x[i+2],1]]
        y_new = [y[i],y[i+1],y[i+2]]
        C = np.linalg.solve(x_new,y_new)
        a[i] = C[0]
        b[i] = C[1]
        c[i] = C[2]
        i += 1
    a[-1] = a[-2]
    b[-1] = b[-2]
    c[-1] = c[-2]
    i = 0
    while i <= n1 - 1:
        j = itr[i]
        if (j == -1):
            y1[i] = nan
            i+=1
            continue
        if ((j>-1) and (j<n)):
            y1[i] = a[j]*x1[i]**2 + b[j]*x1[i]+c[j]
            i+=1
            continue
    return y1

irt = interval(x, x1)
y1 = square_val(x, y, x1, irt)
df = pan.DataFrame({"расчетные ":x1,'значения' :y1})
print(df)
a = min(x)
b = max(x)
x2 = np.arange(a,b,0.1)
itr = interval(x,x2)
y2 = square_val(x,y,x2,itr)
ax.plot(x,y,'ro',x1,y1,'go',x2,y2)
ax.set_title("Кусочно - квадратичная интерполяция")
ax.legend(['Эксперементальные точки', 'Точки вычисления'])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()