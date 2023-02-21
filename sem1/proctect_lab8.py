import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import odeint

def runge_kutta(x0 ,x_end,x, h,):
    n = int(np.ceil((x_end - x0) / h))
    y = np.zeros((n + 1,))
    y[0] = 1

    for k in range(n):
        k1 = y[k]-2*np.sin(x[k])-np.cos(x[k])*(1-2*np.sin(x[k]))
        k2 = f(x + 0.5 * h, y[k] + 0.5 * h * k1) //
        k3 = f(x + 0.5 * h, y[k] + 0.5 * h * k2)
        k4 = f(x + h, y[k] + h * k3)
        y[k + 1] = y[k] + (1 / 6) * h * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h

    return y
def f(x,y):
    y_new = y - 2 * np.sin(x) - np.cos(x) * (1 - 2 * np.sin(x))
    return y_new

h = 0.4
x = np.arange(0, 4, h)
y = runge_kutta(np.min(x),np.max(x),x,h)
df = pan.DataFrame({"расчетные ": x, 'значения': y})
print(df)
