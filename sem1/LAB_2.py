import mplcyberpunk
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
plt.style.use("cyberpunk")
x = np.arange(0, 10, 0.1)
y = np.arange(0, 10, 0.1)
y1 = x ** 3 + x * np.sqrt(x) - x ** 2 - x
y2 = np.cos(x / 2) - x + 5
fig, ax = plt.subplots()
ax.plot(x, y1,'g', x, y2, 'y')
ax.set_title("График зависимости х2 = у(х1)")
ax.set_xlabel("$x1$")
ax.set_ylabel("$x2$")
ax.legend(['y1 = x ** 3 + x * np.sqrt(x) - x ** 2 - x', 'y2 = np.cos(x / 2) - x + 5'])
mplcyberpunk.make_lines_glow(ax,2)
mplcyberpunk.add_gradient_fill(ax,gradient_start='zero')
mplcyberpunk.add_glow_effects()
X_min = 0
X_step = 0.5
X_max = 5
X1, X2 = np.meshgrid(np.arange(X_min, X_max, X_step), y)
F1 = (X1 ** 3) + (X1 * np.sqrt(X1)) - (X1 ** 2) - X2
F2 = np.cos(X1 / 2) - X2 + 5
Z = np.zeros((100, 10))
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_surface(X1, X2, F1)
ax2.plot_surface(X1, X2, F2)
ax2.plot_surface(X1, X2, Z)
mplcyberpunk.make_lines_glow(ax2,1)
mplcyberpunk.add_glow_effects()
fig3 = plt.figure()
ax3 = fig3.add_subplot()
H = np.arange(0, 5, 0.5)
ax3.contour(X1, X2, F1, H)
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.contour(X1, X2, F2, H)
mplcyberpunk.add_glow_effects()
eps = 1e-3
kmax = 50
X_new = np.array([0,0],dtype=float)
D = np.array([0, 0])
X_START = np.array([2, 2], dtype=float)
X_pre = X_START
F = 0
J = 0


def Newton(X_START):
    global F, J
    n = np.size(X_START)
    F = np.array([0, 0],dtype=float)
    J = np.zeros((n, n))
    F[0] = (X_START[0] ** 3) + (X_START[0] * np.sqrt(X_START[0])) - (X_START[0] ** 2) - X_START[1]
    F[1] = np.cos(X_START[0] / 2) - X_START[1] + 5
    J[0, 0] = 3 * X_START[0] ** 2 - 2 * X_START[0] + (3 * np.sqrt(X_START[0])) / 2 - 1
    J[0, 1] = -1
    J[1, 0] = -np.sin(X_START[0] / 2) / 2
    J[1, 1] = -1
    return F, J

def interation(X_START):
    global F, J,X_pre
    n = np.size(X_START)
    F = np.array([0, 0],dtype=float)
    J = np.zeros((n, n),dtype=float)
    F[0] = (X_pre[0] ** 3) + (X_pre[0] * np.sqrt(X_pre[0])) - (X_pre[0] ** 2) - X_pre[1]
    F[1] = np.cos(X_pre[0] / 2) - X_pre[1] + 5
    J[0, 0] = 3 * X_START[0] ** 2 - 2 * X_START[0] + (3 * np.sqrt(X_START[0])) / 2 - 1
    J[0, 1] = -1
    J[1, 0] = -np.sin(X_START[0] / 2) / 2
    J[1, 1] = -1
    return F, J

def fun2(X_pre, F, J):
    global X_new, D
    J1 = inv(J)
    D = np.dot(J1, F)
    X_new = X_pre - D
    return X_new, D


k = 1
Newton(X_START)
fun2(X_START, F, J)
while (np.sum(np.abs(D)) > eps) and (k < kmax):
    X_pre = X_new
    Newton(X_pre)
    fun2(X_pre, F, J)
    k = k + 1
    print("Root:")
    print(X_new)
    print("Number of iterations:")
    print(k)
plt.show()
