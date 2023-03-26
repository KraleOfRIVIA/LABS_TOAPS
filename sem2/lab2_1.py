import numpy as np
import matplotlib.pyplot as plt

fs = 1024 # частота дискретизации
f = 8 # частота сигнала
A = 1 # амплитуда
N = int(fs/f) # число точек в сигнале
t = np.arange(N) # временной интервал
y = A * np.sign(np.sin(2 * np.pi * t * f / fs)) # исходный сигнал
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.plot(t/fs, y)
plt.xlabel('t, с')
plt.ylabel('А')
plt.grid()

fonk = np.sum(y)/N
y = y - fonk
CS = np.fft.fft(y) # частотный спектр сигнала
AS = np.abs(CS) # амплитудный спектр сигнала
plt.subplot(3,2,2)
plt.plot(np.linspace(0, fs, N), AS)
plt.ylabel('Амплитуда')
plt.xlabel('w, Гц')
plt.grid()

# создаем второй частотный спектр
CS1 = CS.copy()
fonk = np.sum(y)/N
y = y - fonk
k = 0
Et = np.linalg.norm(y)**2 # энергия t
Ew = np.linalg.norm(AS)**2 / N # энергия w
while Ew > 0.95 * Et: # продолжаем пока энергия больше 121,6
    CS1[int(N/2) - k] = 0 # зануляем гармоники слева относительно центра (64)
    CS1[int((N/2 + 1)) + k] = 0 # зануляем гармоники справа относительно центра (64)
    Ew = np.linalg.norm(np.abs(CS1))**2 / N # новое значение энергии после зануления
    k += 1
gr = int(N/2) - k
print('Кол-во оставленных гармоник с начала спектра: ', gr)
y1 = np.fft.ifft(CS1)
plt.subplot(3,2,3)
plt.plot(t/fs, y1)
plt.xlabel('t, с')
plt.ylabel('А')
plt.grid()

AS1 = np.abs(CS1)
plt.subplot(3,2,4)
plt.plot(np.linspace(0, fs, N), AS1)
plt.ylabel('Амплитуда')
plt.xlabel('w, Гц')
plt.grid()

FC = np.fft.fftshift(CS)
FC1 = np.fft.fftshift(CS1)
ABS = np.abs(FC)
ABS1 = np.abs(FC1)
plt.subplot(3,2,5)
plt.plot(np.linspace(-fs/2, fs/2, N), ABS)
plt.subplot(3,2,6)
plt.plot(np.linspace(-fs/2, fs/2, N), ABS1)
plt.show()