import numpy as np
import matplotlib.pyplot as plt
f = 8 # частота сигнала
fd = 1024 #частота дискретизации
ph = 0 # начальная фаза
k = fd/f # шаг
A = 1 # амплитуда
t = np.arange(0,k)
y = A * np.sin(2*np.pi*t/k+ph) # синусойда
# y = np.zeros(t.size) для прямоугольного импульса 10 и 11 строка
# y[64:]=1             скаважность 2, если 4, то 64 -> 32
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
ax.plot(t/fd,y)
ax.grid()
ax.set_xlabel('t,c')
ax.set_ylabel('A')
CS = np.fft.fft(y)
AS = np.abs(CS)
FS = np.angle(CS)
ax1.set_xlabel('Сдвиг фазы')
ax1.set_ylabel('w,Гц')
ax1.plot(t*f,FS)
ax1.grid()
fig3, ax3 = plt.subplots()
ax3.stem(t*f,AS)
ax3.set_xlabel('Амплитуда')
ax3.set_ylabel('w , Гц')
ax3.grid()
z = np.fft.ifft(CS)
fig4,ax4 = plt.subplots()
ax4.plot(t/fd,z)
ax4.set_xlabel('t , c')
ax4.set_ylabel('A')
ax4.grid()







plt.show()
