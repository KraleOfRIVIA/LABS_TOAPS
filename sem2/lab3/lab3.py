import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

fd = 1024  # частота дискретизации 
f = 8      # частота сигнала
ph = 0     # начальная фаза
A = 1      # амплитуда
k = fd / f
t = np.arange(k)
const = 4

y = A * np.sin(2 * np.pi * t / k + ph)  # синусоида
y_const = y * const

y1 = A * square(2 * np.pi * t / k, duty=0.25)  # меандр
y1_const = y1 * const

s = np.fft.fft(y)
s_const = np.fft.fft(y_const)
s1 = np.fft.fft(y1)
s1_const = np.fft.fft(y1_const)

AFC = np.abs(s)
AFC_const = np.abs(s_const)
plt.figure(4)
plt.subplot(2, 2, 1)
plt.plot(t / fd, y)
plt.grid(True)
plt.subplot(2, 2, 2)
plt.plot(t * fd / k, AFC, 'r')
plt.grid(True)
plt.subplot(2, 2, 3)
plt.plot(t / fd, y_const)
plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(t * fd / k, AFC_const, 'r')
plt.grid(True)

AFC = np.abs(s1)
AFC_const = np.abs(s1_const)
FFCq1_const = np.angle(s1_const)
FFCq1 = np.angle(s1)
plt.figure(5)
plt.subplot(3, 2, 1)
plt.plot(t / fd, y1, 'r')
plt.grid(True)
plt.subplot(3, 2, 2)
plt.plot(t * fd / k, AFC, 'g')
plt.grid(True)
plt.subplot(3, 2, 3)
plt.plot(t / fd, y1_const, 'r')
plt.grid(True)
plt.subplot(3, 2, 4)
plt.plot(t * fd / k, AFC_const, 'g')
plt.grid(True)
plt.subplot(3, 2, 5)
plt.plot(t * fd / k, FFCq1, 'r')
plt.grid(True)
plt.subplot(3, 2, 6)
plt.plot(t * fd / k, FFCq1_const, 'g')
plt.grid(True)

plt.show()