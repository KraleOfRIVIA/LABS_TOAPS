import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def awgn1(S, SNR):
    n = len(S)               # число отсчетов в сигнале
    Es = np.sum(S**2) / n    # среднеквадратичное значение сигнала
    # SNR = 20 * log10(Es / En)
    En = Es * 10 ** (-SNR / 20)   # среднеквадратичное значение шума
    WGN = np.random.randn(n) * En
    S1 = S + WGN
    return S1

fsig = 1e+3                  # Частота генерации сигнала
N = 100                      # Число отсчетов характеристики + нулевая гармоника
fs = fsig * N                # Частота дискретизации

# Создание прямоугольного сигнала
t = np.arange(N)
y1 = signal.square(2 * np.pi * t / N, duty=0.5)

# Энергия сигнала
signal_energy = np.sum(np.abs(y1) ** 2)

# Наложение AWGN на сигнал с разными уровнями энергии шума
noisy_signal1 = awgn1(y1, SNR=signal_energy)
noisy_signal2 = awgn1(y1, SNR=signal_energy/2)
noisy_signal3 = awgn1(y1, SNR=signal_energy/4)

# Визуализация сигналов
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(t, y1, 'b', linewidth=2)
plt.axis([0, N-1, -1.5, 1.5])
plt.grid(True)
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.title('Исходный прямоугольный сигнал')

plt.subplot(4, 1, 2)
plt.plot(t, noisy_signal1, 'r', linewidth=2)
plt.axis([0, N-1, -1.5, 1.5])
plt.grid(True)
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.title('Сигнал с AWGN (SNR = 0 dB)')

plt.subplot(4, 1, 3)
plt.plot(t, noisy_signal2, 'g', linewidth=2)
plt.axis([0, N-1, -1.5, 1.5])
plt.grid(True)
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.title('Сигнал с AWGN (SNR = 6 dB)')

plt.subplot(4, 1, 4)
plt.plot(t, noisy_signal3, 'm', linewidth=2)
plt.axis([0, N-1, -1.5, 1.5])
plt.grid(True)
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.title('Сигнал с AWGN (SNR = 12 dB)')

plt.tight_layout()
plt.show()
