import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fsig = 1e+3  # Частота генерации сигнала
N = 100  # Число отсчетов характеристики + нулевая гармоника
fs = fsig * N  # Частота дискретизации

# Генерация прямоугольного сигнала
t = np.arange(N)
y1 = signal.square(2 * np.pi * t / N, duty=0.5)

# Массив для сохранения амплитуды пятой гармоники
frequencies = np.linspace(0, fs, 100)
ach_5th = np.zeros_like(frequencies)

# Моделирование прохождения сигнала через линию связи с разными частотами
for i, f in enumerate(frequencies):
    # Применение эффекта линии связи к сигналу
    K = np.exp(-1j * 2 * np.pi * f * t / fs)
    y2 = y1 * K

    # Вычисление спектра и измерение амплитуды пятой гармоники
    Y = np.fft.fft(y2)
    ach_5th[i] = np.abs(Y[5])

# Визуализация зависимости амплитуды пятой гармоники от частоты
plt.figure()
plt.plot(frequencies, ach_5th, linewidth=2)
plt.grid(True)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда пятой гармоники')
plt.title('Амплитуда пятой гармоники в зависимости от частоты прохождения через линию связи')
plt.show()
