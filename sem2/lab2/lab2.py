import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1024 # частота дискретизации
f = 8 # частота сигнала
A = 1 # амплитуда
N = int(fs/f) # число точек в сигнале
t = np.arange(N) # временной интервал
y = A * signal.square(2 * np.pi * f * t / fs,0.25) # импульс скважности 2
plt.figure(1)

plt.subplot(3, 2, 1) # график размещаем
plt.plot(t/fs, y) # наш исходный график (частотный)
plt.xlabel('t, с')
plt.ylabel('А')
plt.grid(True)

CS = np.fft.fft(y) # ЧАСТОТНЫЙ СПЕКТР СИГНАЛА
AS = np.abs(CS) # АМПЛИТУДНЫЙ СПЕКТР СИГНАЛА

plt.subplot(3, 2, 2)
plt.plot(np.arange(N)/N * fs, AS)
plt.ylabel('Амплитуда')
plt.xlabel('w, Гц')
plt.grid(True)

# создаем второй частотный спектр
CS1 = CS.copy()
k = 0
gr = 8
while k < (N//2-gr):
    CS1[N//2-k] = 0
    CS1[(N//2+1)+k] = 0
    k += 1

y1 = np.fft.ifft(CS1)

plt.subplot(3, 2, 3)
plt.plot(t/fs, y1) # ВТОРОЙ ГРАФИК ЧАСТОТНЫЙ
plt.xlabel('t, с')
plt.ylabel('А')
plt.grid(True)

AS1 = np.abs(CS1)

plt.subplot(3, 2, 4)
plt.plot(np.arange(N)/N * fs, AS1)
plt.ylabel('Амплитуда')
plt.xlabel('w, Гц')
plt.grid(True)

plt.show()