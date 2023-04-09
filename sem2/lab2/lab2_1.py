import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1024 # частота дискретизации
f = 8 # частота сигнала
A = 1 # амплитуда
N = int(fs/f) # число точек в сигнале
t = np.arange(N) # временной интервал
y = A * signal.square(2 * np.pi * t/N,0.5) # импульс скважности 2
plt.figure(1)
plt.subplot(3, 2, 1) # график размещаем
plt.plot(t/fs, y) # наш исходный график (частотный)
plt.xlabel('t, с')
plt.ylabel('А')
plt.grid(True)
fonk = np.sum(y)/128
y = y - fonk
CS = np.fft.fft(y) # ЧАСТОТНЫЙ СПЕКТР СИГНАЛА
AS = np.abs(CS) # АМПЛИТУДНЫЙ СПЕКТР СИГНАЛА
plt.subplot(3, 2, 2)
plt.plot(t/fs, AS)
plt.ylabel('Амплитуда')
plt.xlabel('w, Гц')
plt.grid(True)
# создаем второй частотный спектр
CS1 = CS.copy()
fonk = np.sum(y)/128
y = y - fonk
k = 0
Et = np.linalg.norm(y)**2 # энергия t
Ew = np.linalg.norm(AS)**2 / N # энергия w
while Ew > 0.95 * Et: # продолжаем пока энергия больше 121,6
    CS1[int(N/2) - k] = 0 # зануляем гармоники слева относ. центра (64)
    CS1[int((N/2 + 1) + k)] = 0 # зануляем гармоники справа относ. центра (64)
    Ew = np.linalg.norm(np.abs(CS1))**2 / N # новое значение энергии после зануления
    k += 1
    gr = int(N/2 - k)
    print('Кол-во оставленных гармоник с начала спектра: ')
    print(gr)
y1 = np.fft.ifft(CS1)
plt.subplot(3, 2, 3)
plt.plot(t/fs, y1) # ВТОРОЙ ГРАФИК ЧАСТОТНЫЙ
plt.xlabel('t, с')
plt.ylabel('А')
plt.grid(True)
AS1 = np.abs(CS1)
plt.subplot(3, 2, 4)
plt.plot(t/fs, AS1)
plt.ylabel('Амплитуда')
plt.xlabel('w, Гц')
plt.grid(True)
FC = np.fft.fftshift(CS)
FC1 = np.fft.fftshift(CS1)
ABS = np.abs(FC)
ABS1 = np.abs(FC1)
plt.subplot(3, 2, 5)
plt.plot(t*f,ABS)
plt.subplot(3,2,6)
plt.plot(t*f,ABS1)

plt.show()