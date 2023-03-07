import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import odeint
import pandas as pan

def runge_kutta(x, h,):
    n = np.size(x)
    y = np.zeros((n))
    y[0] = 1

    for k in range(0,len(x)-1):
        k1 =  y[k]-2*np.sin(x[k])-np.cos(x[k])*(1-2*np.sin(x[k]))
        k2 =  (y[k]+ 0.5 * h * k1)-2*np.sin(x[k]+ 0.5 * h)-np.cos(x[k]+ 0.5 * h)*(1-2*np.sin(x[k]+ 0.5 * h))
        k3 =  (y[k]+ 0.5 * h * k2)-2*np.sin(x[k]+ 0.5 * h)-np.cos(x[k]+ 0.5 * h)*(1-2*np.sin(x[k]+ 0.5 * h))
        k4 =  (y[k]+ 0.5 * h * k3)-2*np.sin(x[k]+ h)-np.cos(x[k]+ h)*(1-2*np.sin(x[k]+ h))
        y[k + 1] = y[k] + (1 / 6) * h * (k1 + 2 * k2 + 2 * k3 + k4)

    return y
def f(x,y):
    y_new = y - 2 * np.sin(x) - np.cos(x) * (1 - 2 * np.sin(x))
    return y_new

i = f(1,1)
h = 0.4
x = np.arange(0, 4.4, h)
y = runge_kutta(x,h)
df = pan.DataFrame({"расчетные ": x, 'значения': y})
print(df)
fig, ax = plt.subplots()
ax.plot(x,y)
ax.set_title("Рунге-Кутты")
ax.legend(['Эксперементальные точки', 'Точки вычисления'])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()
# f(x[k], y[k])[0]
# f(x[k] + 0.5 * h, y[k] + 0.5 * h * k1)[0]
# f(x[k] + 0.5 * h, y[k] + 0.5 * h * k2)[0]
# f(x[k] + h, y[k] + h * k3)[0]
# f(x[k] + h, y[k] + h * k3)[0] //