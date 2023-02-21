import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-10,10,1)
dx = x[1]-x[0]
y = (-1 * np.arctan(x) - 1.2)*x
y1 = -x/(1+x**2)-np.arctan(x)-6/5
dydx = np.gradient(y, dx)
fig, ax = plt.subplots()
ax.plot(x,dydx,linewidth = "2.0")
plt.plot(x,y1, color = 'r')
plt.grid()
plt.show()
