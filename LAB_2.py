import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mplot3d_dragger import Dragger3D

x = np.arange(0, 10, 0.1)
y = np.arange(0, 10, 0.1)
y1 = x ** 3 + x * np.sqrt(x) - x ** 2 - x
y2 = np.cos(x / 2) - x + 5
fig, ax = plt.subplots()
ax.plot(x, y1, 'b', x, y2, 'r')
ax.set_title("График зависимости х2 = у(х1)")
ax.set_xlabel("$x1$")
ax.set_ylabel("$x2$")
plt.grid()
X_min = 0
X_step = 0.5
X_max = 5
X1, X2 = np.meshgrid(np.arange(X_min, X_max, X_step), y)
F1 = (X1 ** 3) + (X1 * np.sqrt(X1)) - (X1 ** 2) - X2
F2 = np.cos(X1 / 2) - X2 + 5
Z = np.zeros((100, 10))
fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# dr = Dragger3D(ax2)
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_surface(X1, X2, F1)
ax2.plot_surface(X1, X2, F2)
ax2.plot_surface(X1, X2, Z)
fig3 = plt.figure()
ax3 = fig3.add_subplot()
H = np.arange(0, 5, 0.5)
ax3.contour(X1, X2, F1, H)
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.contour(X1, X2, F2, H)
plt.grid()
eps = 0.001
kmax = 50
X_new = 0
X = np.array([2, 2])
D = 0
F= 0
J = 0
def fun1(X):
    global F,J
    n = np.size(X)
    F = np.zeros(n)
    J = np.zeros((n,n))
    F[0] = X[0] ** 3 + X[0] * np.sqrt(X[0]) - X[0] ** 2 - X[0]
    F[1] = np.cos(X[0] / 2) - X[0] + 5
    J[0,0] = 3 * X[0] ** 2 - 2 * X[0] + (3 * np.sqrt(X[0])) / 2 - 1
    J[0, 1] = 1
    J[1,0] = -np.sin(x[0] / 2) / 2 - 1
    J[1, 1] = 1
    return F, J


def fun2(X, F, J):
    D = -1 * (J / F)
    X_new = X[0] + D
    return X_new, D


k = 1
fun1(X)
fun2(X, F, J)

while np.max(np.abs(D)) and (k < kmax):
    X = X_new
    fun1(X)
    fun2(X, F, J)
    k = k + 1
    print(X_new)
print("Root:")
print(X_new)
print("Number of iterations:")
print(k)
plt.show()
