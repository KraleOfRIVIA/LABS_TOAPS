import numpy as np
import matplotlib.pyplot as plt

fd = 1024  # частота дискретизации
f = 8  # частота сигнала
ph = 0
A = 1  # амплитуда
k = fd / f
t = np.arange(k)

const = 4

y = A * np.sign(np.sin(2 * np.pi * t / k))
y_const = np.roll(y, 8)

y1 = A * np.sign(np.sin(2 * np.pi * t / k + np.pi / 2))
y1_const = np.roll(y1, 8)

# преобразование Фурье
s = np.fft.fft(y)
s_const = np.fft.fft(y_const)
s1 = np.fft.fft(y1)
s1_const = np.fft.fft(y1_const)

AFC = np.abs(s)
AFC_const = np.abs(s_const)
FFCq_const = np.angle(s_const)
FFCq = np.angle(s)

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
axs[0].plot(t / fd, y, 'r')
axs[0].plot(t / fd, y_const, 'g')
axs[0].grid(True)

axs[1].plot(t * fd / k, AFC_const, 'g')
axs[1].plot(t * fd / k, AFC, 'r')
axs[1].grid(True)

axs[2].plot(t * fd / k, FFCq_const, 'g')
axs[2].plot(t * fd / k, FFCq, 'r')
axs[2].grid(True)

y1 = A * np.sign(np.sin(2 * np.pi * t / k))
y1_const = np.roll(y1, 8)

# 5 гармоник не равны 0 для меандра без скважности

AFC = np.abs(s1)
AFC_const = np.abs(s1_const)
FFCq1_const = np.angle(s1_const)
FFCq1 = np.angle(s1)

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
axs[0].plot(t / fd, y1, 'g')
axs[0].plot(t / fd, y1_const, 'r')
axs[0].grid(True)

axs[1].plot(t * fd / k, AFC, 'r')
axs[1].plot(t * fd / k, AFC_const, 'g')
axs[1].grid(True)

axs[2].plot(t * fd / k, FFCq1, 'g')
axs[2].plot(t * fd / k, FFCq1_const, 'r')
axs[2].grid(True)

plt.show()