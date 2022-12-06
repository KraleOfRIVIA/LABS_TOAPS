import numpy as np
import matplotlib.pyplot as plt
from math import nan
import pandas as pan
# x=[0.5, 3.5, 6.5, 8, 9.5, 11, 13.5, 16.5, 18.5]
# y=[2.5, 13, 10.5, 2, 1.5, 6, 9, 4.5, 3.5]
# x1=[1, 3, 7.5, 9, 12, 15 ,20]
# x=[-9, -7, -4 ,-2.5 ,-1.5 ,1, 2.5, 3.5 ,5, 5.5]
# y=[-5 ,-2.75, -2, -2.5, -3 ,-4.5, -4 ,-2.75, 2.5 ,8]
# x1=[-8, -5, -0.5 ,2 ,3 ,4 ,4.5, 5.25]
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
def progom(x, y):
    n = np.size(x) - 1
    h = np.arange(0, n, 1.0)
    h[0:n] = x[1:n + 1] - x[0:n]
    A = np.zeros((n),dtype=float)
    A[1:n - 1] = h[1:n - 1]
    B = np.zeros((n),dtype=float)
    B[0:n - 1] = 2 * (h[0:n - 1] + h[1:n])
    C = np.zeros((n))
    C[0:n - 2] = h[1:n - 1]
    C[n - 1] = 0
    D = np.zeros((n-1),dtype=float)
    for i in range(0,n-1):
        D[i] = 6*((y[i+2]-y[i+1])/h[i+1]-(y[i+1]-y[i])/h[i])
    Q = np.zeros((n),dtype=float)
    R = np.zeros((n+1),dtype=float)
    for i in range(0,n-1):
        Q[i+1] = -(C[i]/(B[i]+A[i]*Q[i]))
        R[i+1] = (D[i]-A[i]*R[i])/(B[i]+A[i]*Q[i])
    M = np.zeros((n),dtype=float)
    M[n-1] = R[n]
    for i in range(n-2,0,-1):
        M[i] = Q[i+1]*M[i+1]+R[i+1]
    M = np.insert(M,0,0)
    M = np.append(M,[0])
    return M

def interval(x, x1):
    n = np.size(x)
    n1 = np.size(x1)
    itr = np.zeros(n1,dtype=int)
    for i in range(0, n1):
        if (x1[i] < x[0]):
            itr[i] = -1
            continue
        if (x1[i] > x[-1]):
            itr[i] = n
        j = 0
        while (j <= n - 2):
            if (x1[i] >= x[j]) and (x1[i] <= x[j + 1]):
                itr[i] = j
                break
            else:
                j += 1
    # itr[-1] = itr[-2]
    return itr

def spline_val(x,y,x1,itr,M):
    n = np.size(x) -1
    n1 = np.size(x1)
    y1 = np.zeros((n1),dtype=float)
    h = np.arange(0,n+1,1.0)
    h[0:n] = x[1:n+1] - x[0:n]
    i = 0
    while(i<=n1-1):
        j = itr[i]
        if (j==-1):
            y1[i] = y[0]+((x[0]-x[1])*M[4]/6+(y[1]-y[0])/(x[1]-x[0]))*(x1[i]-x[0])
            i+=1
            # continue
        if (j>-1 and j<=n):
            y1[i] =1/((6*h[j]))*((M[j]*(x[j+1] - x1[i])**3)+M[j+1]*(x1[i]-x[j])**3)+(1/h[j])*((y[j]-((M[j]*h[j]**2)/6))*(x[j+1]-x1[i])+(y[j+1]-((M[j+1]*h[j]**2)/6))*(x1[i]-(x[j])))
            i+=1
            # continue
        if (j>=n):
            y1[i] = y[-1]+((x[-1]-x[-2])*M[n]/6+(y[-1]-y[-2])/(x[-1]-x[-2]))*(x1[i]-x[-1])
            i+=1
            # continue
    return y1
M = progom(x,y)
itr = interval(x,x1)
y1 = spline_val(x,y,x1,itr,M)
df = pan.DataFrame({"расчетные ":x1,'значения' :y1})
print(df)
if min(x) < min(x1):
    a = min(x)
else:
    a = min(x1)
if max(x) > max(x1):
    b = max(x)
else:
    b = max(x1)
x2 = np.arange(a,b,0.01)
itr = interval(x,x2)
y2 = spline_val(x,y,x2,itr,M)
df = pan.DataFrame({"расчетные ":x2,'значения' :y2})
print(df)
fig, ax = plt.subplots()
ax.plot(x,y,'ro',x1,y1,'go',x2,y2)
ax.set_title("Кубический сплайн")
ax.legend(['Эксперементальные точки', 'Точки вычисления'])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()
