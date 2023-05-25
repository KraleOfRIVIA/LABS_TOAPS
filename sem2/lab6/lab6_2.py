import numpy as np
import matplotlib.pyplot as plt


fsig = 1e3
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
plt.semilogx(f, ACH, linewidth=2)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('АЧХ линии связи')

plt.subplot(212)
plt.semilogx(f, FCH, linewidth=2)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('angle(f)')
plt.title('ФЧХ линии связи')


A = 1
k = len(f)
t = np.arange(0, k)
y1 = A * np.sign(np.sin(2 * np.pi * t / k)) # прямоугольный сигнал

plt.figure(2)

plt.subplot(121)
plt.plot(t, y1, '-b', linewidth=2)
plt.axis([0, 100, -1.5, 1.5])
plt.grid(True)
plt.xlabel('N, номер отсчета')
plt.ylabel('y1(N)')
plt.title('Исходный сигнал во временной области')

S1 = np.fft.fft(y1) # Комплексный спектр исходного сигнала
ACH_S1 = np.abs(S1) # АЧХ
FCH_S1 = np.unwrap(np.angle(S1)) # ФЧХ

plt.figure(3)

plt.subplot(221)
plt.plot(f, ACH_S1)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('АЧХ исходного сигнала')

plt.subplot(222)
plt.plot(f, FCH_S1)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('angle(f)')
plt.title('ФЧХ исходного сигнала')


S1[(N//2 + 2):] = 0 # зануляем мнимую часть спектра сигнала

S2 = S1 * K1
ACH_S2 = np.abs(S2) # АЧХ
FCH_S2 = np.unwrap(np.angle(S2)) # ФЧХ

plt.subplot(223)
plt.stem(f, ACH_S2)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('АЧХ принятого сигнала')

plt.subplot(224)
plt.plot(f, FCH_S2)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('angle(f)')
plt.title('ФЧХ принятого сигнала')


y2 = 2 * np.fft.ifft(S2)

plt.figure(2)

plt.subplot(122)
plt.plot(t, np.real(y2), '-r', linewidth=2)
plt.grid(True)
plt.xlabel('N, номер отсчета')
plt.ylabel('y2(N)')
plt.title('Сигнал после линии связи')

plt.show()