import numpy as np
import matplotlib.pyplot as plt
x1 = 0
D = 0
x = np.arange(-10,10,0.1)
y = (-1 * np.arctan(x) - 1.2)*x
y1 = -x/(1+x**2)-np.arctan(x)-6/5
y2 = x
F1 = 0
F = 0
M = -1
fig, ax, = plt.subplots()
plt.plot(x,y1, color = 'r')
plt.plot(x,y2, color = 'g')
ax.plot(x,y,linewidth = "2.0")
ax.set_title("График функции y = f(x)")
ax.set_xlabel("$x$")
ax.legend(['y = -x/(1+x^2)-arctan(x)-6/5', 'y = х','y = (– arctg x – 1,2) x'])
ax.set_ylabel("$y$")
plt.grid()
eps = 1e-3
kmax = 50
x = -1
k = 1
def fun1(x):
    global F,F1
    F = (-1 * np.arctan(x) - 1.2)*x
    F1 = x + F/M
    return F,F1
def fun2(*args):
    global x1,F,F1, M,D
    D = (-1 * F) / F1
    x1 = x - D
fun1(x)
fun2(x,F,F1)
print(k,x1,D)
while (np.abs(D)>eps) and (k<kmax):
    x = x1
    fun1(x)
    fun2(x,F,F1)
    k = k+1
    print(k, x1, D)
print('Root: x =', x1,'\n')
print('Number of iterations: k =',k,'\n')
print('Accuracy: D =', D,'\n')

plt.show()