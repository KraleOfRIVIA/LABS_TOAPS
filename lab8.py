import numpy as np
import matplotlib.pyplot as plt
import pandas as pan

def progom(x, y):
    n = np.size(x) - 1
    h = np.arange(0, n, 1.0)
    h[0:n] = x[1:n + 1] - x[0:n]
    A = np.zeros((n), dtype=float)
    A[1:n - 1] = h[1:n - 1]
    B = np.zeros((n), dtype=float)
    B[0:n - 1] = 2 * (h[0:n - 1] + h[1:n])
    C = np.zeros((n))
    C[0:n - 2] = h[1:n - 1]
    C[n - 1] = 0
    D = np.zeros((n - 1), dtype=float)
    for i in range(0, n - 1):
        D[i] = 6 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])
    Q = np.zeros((n), dtype=float)
    R = np.zeros((n + 1), dtype=float)
    for i in range(0, n - 1):
        Q[i + 1] = -(C[i] / (B[i] + A[i] * Q[i]))
        R[i + 1] = (D[i] - A[i] * R[i]) / (B[i] + A[i] * Q[i])
    M = np.zeros((n), dtype=float)
    M[n - 1] = R[n]
    for i in range(n - 2, 0, -1):
        M[i] = Q[i + 1] * M[i + 1] + R[i + 1]
    M = np.insert(M, 0, 0)
    M = np.append(M, [0])
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
            y1[i] = y[0]+((x[0]-x[1])*M[3]/6+(y[1]-y[0])/(x[1]-x[0]))*(x1[i]-x[0])
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

def Eiler_Koshi(x,h):
    n = np.size(x)
    y = np.zeros((n))
    y1 = np.zeros((n))
    x1 = np.zeros((n))
    y[0] = 1
    for i in range(0, len(x)-1):
        y1[i+1] = y[i]+h*(y[i]-2*np.sin(x[i])-np.cos(x[i])*(1-2*np.sin(x[i])))
        x1[i] = x[i] +h
        y[i + 1] = y[i]+h/2*((y[i]-2*np.sin(x[i])-np.cos(x[i])*(1-2*np.sin(x[i])))+(y1[i]-2*np.sin(x1[i])-np.cos(x1[i])*(1-2*np.sin(x1[i])))+h)
    return y



h = 0.4
x = np.arange(0, 4, h)
y = Eiler_Koshi(x, h)
df = pan.DataFrame({"расчетные ": x, 'значения': y})
print(df)
M = progom(x, y)
a = np.min(x)
b = np.max(x)
x2 = np.arange(a, b, 0.1*h)
itr = interval(x, x2)
y2 = spline_val(x, y, x2, itr, M)
fig, ax = plt.subplots()
ax.plot(x, y, 'ro', x2, y2)
ax.set_title("Метод Эйлера-Коши")
ax.legend(['Эксперементальные точки', 'Точки вычисления'])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()