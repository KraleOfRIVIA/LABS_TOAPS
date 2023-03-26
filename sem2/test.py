import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
T = 1 # Период
duty_cycles = np.linspace(0.01, 0.9, 100) # Различные значения скважности

# Вычисляем спектр для каждой скважности
spectra = []
for duty_cycle in duty_cycles:
    t = np.linspace(0, T, 1000)
    x = np.zeros_like(t)
    x[t < duty_cycle*T] = 1
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), t[1] - t[0])
    spectra.append(np.abs(X))

# Определяем минимальную практическую ширину спектра
spectra = np.array(spectra)
spectra_sum = spectra.sum(axis=1)
min_idx = np.argmin(spectra_sum)
min_duty_cycle = duty_cycles[min_idx]

# Выводим график спектра и минимальную скважность
plt.plot(freqs, spectra[min_idx])
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title(f'Minimum spectral width at duty cycle {min_duty_cycle:.2f}')
plt.show()