import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
# x=[-8, -6 ,-3.5, -3, -2.5, 0, 2, 2.5, 4, 6.5]
# y=[-1 ,3 ,6.5 ,4 ,2 ,4 ,4.5, 1, -2, 1]
# x1=[-10 ,-9 ,-5 ,-1, 1.5, 3, 5, 9]
x = [-9.5, -6.5, -4, -2.5, -0.5, 1.5, 3, 4.5, 9.5]
y = [5.5, 1, -4.5, -2, -5, -2.5, 0, 1.5, -1.5]
x1 = [-8, -5, -3, -1.5, 0.5, 4, 8]
cord_x_and_y = np.column_stack([x, y])
sort = np.argsort(cord_x_and_y[:, 0])
cord_x_and_y = cord_x_and_y[sort]
x1.sort()
[x, y] = np.hsplit(cord_x_and_y, 2)
x = np.hstack(x)
y = np.hstack(y)
def lagr_val(x,y,x1):
    n = np.size(x) - 1
    n1 = np.size(x1)
    y1 = np.zeros((n1))
    k = 0
    while(k<=n1-1):
        if (x1[k]<x[0]) or (x1[k]>x[n]):
            y1[k] = 0
            k+=1
        else:
            i = 0
            while(i<= n):
                P = 1
                j = 0
                while(j<=n):
                    if (i != j):
                        P = P * ((x1[k]-x[j])/(x[i]-x[j]))
                    j +=1
                y1[k] = y1[k]+P*y[i]
                i += 1
            k+=1
    return y1

y1 = lagr_val(x,y,x1)
df = pan.DataFrame({"расчетные ":x1,'значения' :y1})
print(df)
a = np.min(x)
b = np.max(x)
x2 =np.arange(a,b,0.001)
y2 = lagr_val(x,y,x2)
fig, ax = plt.subplots()
ax.plot(x,y,'ro',x1,y1,'go',x2,y2)
ax.legend(['Эксперементальные точки', 'Точки вычисления'])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()