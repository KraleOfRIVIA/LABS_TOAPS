import numpy as np
import matplotlib.pyplot as plt

# Сигнал с наименьшей шириной спектра - синусоида
t = np.linspace(0, 1, 1000) # время
f = 10 # частота сигнала
s = np.sin(2*np.pi*f*t) # сигнал

# Спектр сигнала
S = np.fft.fft(s)
freq = np.fft.fftfreq(s.shape[-1], d=t[1]-t[0])

# Визуализация сигнала и его спектра
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(t, s)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Signal with narrowest spectrum')

ax[1].plot(freq, np.abs(S))
ax[1].set_xlim(-50, 50) # Ограничиваем диапазон частот для лучшей визуализации
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Magnitude')
ax[1].set_title('Spectrum')

plt.show()

T = 1.0  # ширина импульса
N = 1024  # число отсчетов
t = np.linspace(-T/2, T/2, N)  # временная шкала

# построение сигнала
rect = np.zeros(N)
rect[np.abs(t) < T/2] = 1

# построение спектра сигнала
spectrum = np.fft.fft(rect)

# построение графиков
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(t, rect)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Signal with narrowest spectrum')

freq = np.fft.fftfreq(N, t[1]-t[0])
ax2.plot(freq, np.abs(spectrum))
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_xlim(-100, 100)
ax2.set_title('Spectrum')

plt.tight_layout()
plt.show()