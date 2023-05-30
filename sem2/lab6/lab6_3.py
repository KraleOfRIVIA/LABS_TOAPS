import numpy as np
import matplotlib.pyplot as plt

fsig = 1*np.e+3  # Частота прямоугольного сигнала в Гц
N = 100  # Количество отсчетов
fs = fsig * N  # Частота дискретизации

t = np.arange(0, N) / fs  # Временная шкала
f0 = fsig  # Частота прямоугольного сигнала

signal = np.zeros(N)  # Создаем массив для прямоугольного сигнала

# Генерируем прямоугольный сигнал
signal[:N//2] = 1

# Моделируем прохождение через линию связи с затуханием пятой гармоники
attenuation_factor = 0.5  # Фактор затухания для пятой гармоники
f5 = 5 * f0  # Частота пятой гармоники
print('Частота 5-ой гармоники',f5)
signal_attenuated = signal.copy()
signal_attenuated[int(N * f5 / fs):] *= attenuation_factor

# Выводим графики сигнала и затухшей пятой гармоники
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Прямоугольный сигнал')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда')

plt.subplot(2, 1, 2)
plt.plot(t, signal_attenuated)
plt.title('Затухшая пятая гармоника')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда')

plt.tight_layout()
plt.show()
