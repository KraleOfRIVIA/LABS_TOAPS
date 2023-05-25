import numpy as np
import matplotlib.pyplot as plt

fsig = 1e4
N = 100
fs = fsig * N
f = np.arange(0, fs, fsig)

l = 1000
R = 5e-3 + (42e-3) * np.sqrt(f * (1e-6))
L = 2.7e-7
G = 20 * f * (1e-15)
C = 48e-12

w = 2 * np.pi * f
g1 = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))
K1 = np.exp(-g1 * l)

ACH = np.abs(K1)
FCH = np.unwrap(np.angle(K1))

plt.figure(1)

plt.subplot(211)
plt.semilogx(f, ACH)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('АЧХ линии связи')

plt.subplot(212)
plt.semilogx(f, FCH)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('angle(f)')
plt.title('ФЧХ линии связи')

plt.show()