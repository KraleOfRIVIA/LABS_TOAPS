import numpy as np
import matplotlib.pyplot as plt

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
D = 0.0
X_STAR = np.array([2.0, 2.0], dtype=float)
# n = np.size(X1)
# F = np.array([0.0, 0.0])
# J = np.zeros((n, n), dtype=float)
F = 0
J = 0
def fun1(X_STAR):
    global F,J
    n = np.size(X_STAR)
    F = np.array([0,0])
    J = np.zeros((n, n), dtype=float)
    F[0] = (X_STAR[0] ** 3) + (X_STAR[0] * np.sqrt(X_STAR[0])) - (X_STAR[0] ** 2) - X_STAR[1]
    F[1] = np.cos(X_STAR[0] / 2) - X_STAR[1] + 5
    J[0, 0] = 3 * X_STAR[0] ** 2 - 2 * X_STAR[0] + (3 * np.sqrt(X_STAR[0])) / 2 - 1
    J[0, 1] = 1
    J[1, 0] = -np.sin(X_STAR[0] / 2) / 2 - 1
    J[1, 1] = 1
    return F, J

def fun2(X_STAR,F,J):
    J = J.T
    D = -1 *(J/F)
    X_new = X_STAR[0] + D
    return X_new, D


k = 1
fun1(X_STAR)
fun2(X_STAR,F,J)
print(D)
while (np.max(np.abs(D)) > eps) and (k < kmax):
    X_STAR = X_new
    fun1(X_STAR)
    fun2(X_STAR,F,J)
    k = k + 1
    print(D)
    print(X_new)
print("Root:")
print(X_new)
print("Number of iterations:")
print(k)
plt.show()
