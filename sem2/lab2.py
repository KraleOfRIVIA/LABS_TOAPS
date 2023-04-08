import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1024 # частота дискретизации
f = 8 # частота сигнала
A = 1 # амплитуда
N = int(fs/f) # число точек в сигнале
t = np.arange(N) # временной интервал
y = A * np.sign(np.sin(2 * np.pi * t * f / fs)) # импульс скважности 2

fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(t/fs, y)
axs[0, 0].set_xlabel('t, с')
axs[0, 0].set_ylabel('А')
axs[0, 0].grid(True)
AS = np.abs(np.fft.fft(y))
axs[0, 1].plot(np.fft.fftfreq(N, 1/fs), AS)
axs[0, 1].set_xlabel('w, Гц')
axs[0, 1].set_ylabel('Амплитуда')
axs[0, 1].grid(True)
CS1 = np.fft.fft(y)
gr = 8
CS1[int(N/2-gr):int(N/2+gr)+1] = 0
y1 = np.fft.ifft(CS1)
axs[1, 0].plot(t/fs, y1)
axs[1, 0].set_xlabel('t, с')
axs[1, 0].set_ylabel('А')
axs[1, 0].grid(True)
AS1 = np.abs(CS1)
axs[1, 1].plot(np.fft.fftfreq(N, 1/fs), AS1)
axs[1, 1].set_xlabel('w, Гц')
axs[1, 1].set_ylabel('Амплитуда')
axs[1, 1].grid(True)
FC = np.fft.fftshift(np.fft.fft(y))
ABS = np.abs(FC)
axs[2, 0].plot(np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) * f, ABS)
axs[2, 0].set_xlabel('f, Гц')
axs[2, 0].set_ylabel('Амплитуда')
axs[2, 0].grid(True)
FC1 = np.fft.fftshift(CS1)
ABS1 = np.abs(FC1)
axs[2, 1].plot(np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) * f, ABS1)
axs[2, 1].set_xlabel('f, Гц')
axs[2, 1].set_ylabel('Амплитуда')
axs[2, 1].grid(True)

plt.tight_layout()
plt.show()

