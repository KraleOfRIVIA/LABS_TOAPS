import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mplot3d_dragger import Dragger3D
x = np.arange(0,10,0.1)
y = np.arange(0,10,0.1)
y1 = x**3+x*np.sqrt(x)-x**2-x
y2 = np.cos(x/2)-x+5
fig, ax = plt.subplots()
ax.plot(x,y1,'g',x,y2,'r')
ax.set_title("График зависимости х2 = у(х1)")
ax.set_xlabel("$x1$")
ax.set_ylabel("$x2$")
plt.grid()
X_min = 0
X_step = 0.25
X_max = 5
X1, X2 = np.meshgrid(np.arange(X_min,X_max,X_step),y)
F1 = (X1**3)+(X1*np.sqrt(X1))-(X1**2)-X2
F2 = np.cos(X1/2)-X2+5
Z = np.zeros(np.size(X1))
fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# dr = Dragger3D(ax2)
plt.show()

