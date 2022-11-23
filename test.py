import numpy as np
import matplotlib.pyplot as plt
from math import nan, pi
import pandas as pan
fig, ax = plt.subplots()
x = np.arange(-pi,pi+1,pi/2)
y = np.cos(x)
x1 = np.arange(np.min(x),np.max(x),0.1)
m = np.size(x) - np.size(x1)
def interval(x,x1):
    global m
    k = np.size(x) - m
    n = np.size(x) - 1
    n1 = np.size(x1)
    irt = np.zeros((n1))
    for i in range(0, n1):
        if (x1[i] < x[0]):
            irt[i] = -1
        if (x1[i]>x[0]and x1[i]<x[-1]):
            j = 1
            while (x1[i]>x[j]):
                j+=1
                if j == n:

                    break
                if j == k:
                    j = k
                    break
            irt[i] = j - 1


    return irt

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
irt = irt.astype(np.int64)
irt = np.hstack(irt)
y1 = square_val(x, y, x1, irt)
df = pan.DataFrame({"расчетные ":x1,'значения' :y1})
print(df)
a = min(x)
b = max(x)
x2 = np.arange(a,b,0.01)
itr = interval(x,x2)
itr = itr.astype(np.int64)
itr = np.hstack(itr)
y2 = square_val(x,y,x2,itr)
ax.plot(x,y,'ro',x2,y2)
ax.set_title("Кусочно - квадратичная интерполяция")
ax.legend(['Эксперементальные точки', 'Точки вычисления'])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()
