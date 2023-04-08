import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fd = 1024  # частота дискретизации
f = 8  # частота сигнала
ph = 0  # начальная фаза
A = 1  # амплитуда
k = fd / f
t = np.arange(k)
const = 2

y = A * np.sin(2 * np.pi * t / k + ph)  # синусоида
y_const = y * const
y_super = y + y_const

y1 = A * signal.square(2 * np.pi * t / k, duty=0.25)  # меандр
y1_const = A * signal.square(2 * np.pi * t / k, duty=0.125)
y1_super = y1 + y1_const

s = np.fft.fft(y)
s_const = np.fft.fft(y_const)
s_super = np.fft.fft(y_super)
s1 = np.fft.fft(y1)
s1_const = np.fft.fft(y1_const)
s1_super = np.fft.fft(y1_super)

AFC = np.abs(s)
AFC_const = np.abs(s_const)
AFC_super = np.abs(s_super)
plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(t / fd, y)
plt.plot(t / fd, y_const, 'g')
plt.plot(t / fd, y_super, 'r')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(t * fd / k, AFC_super)
plt.plot(t * fd / k, AFC_const, 'g')
plt.plot(t * fd / k, AFC, 'r')
plt.grid(True)

AFC = np.abs(s1)
AFC_const = np.abs(s1_const)
AFC_super = np.abs(s1_super)
FFCq1_const = np.angle(s1_const)
FFCq1 = np.angle(s1)
FFCq1_super = np.angle(s1_super)
plt.figure(5)
plt.subplot(3, 1, 1)
plt.plot(t / fd, y1)
plt.plot(t / fd, y1_const, 'r')
plt.plot(t / fd, y1_super, 'g')
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(t * fd / k, AFC)
plt.plot(t * fd / k, AFC_const, 'r')
plt.plot(t * fd / k, AFC_super, 'k')
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(t * fd / k, FFCq1_super, 'g')
plt.plot(t * fd / k, FFCq1_const, 'r')
plt.plot(t * fd / k, FFCq1, 'b')
plt.grid(True)

plt.show()