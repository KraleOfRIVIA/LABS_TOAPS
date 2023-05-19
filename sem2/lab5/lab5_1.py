import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import correlate

# количество точек
N = 1000

# Белый шум с равномерным распределением
uniform_noise = np.random.uniform(-1, 1, N)

# Аддитивный белый гауссовский шум
gaussian_noise = np.random.normal(0, 1, N)

def plot_noise(noise, title):
    # Сигнал шума
    plt.figure(figsize=(14, 6))
    plt.plot(noise)
    plt.title(f'{title} сигнал')
    plt.show()

    # Спектр
    yf = fft(noise)
    xf = fftfreq(N, 1)
    plt.figure(figsize=(14, 6))
    plt.plot(xf, np.abs(yf))
    plt.title(f'{title} спектр')
    plt.show()

    # Распределение во временной области
    plt.figure(figsize=(14, 6))
    plt.hist(noise, bins=50)
    plt.title(f'{title} распределение во временной области')
    plt.show()

    # Распределение в спектральной области
    powerSpectralDensity = np.abs(yf) ** 2
    plt.figure(figsize=(14, 6))
    plt.plot(xf, powerSpectralDensity)
    plt.title(f'{title} распределение в спектральной области')
    plt.show()

    # Автокорреляционная функция
    autocorr = correlate(noise, noise)
    plt.figure(figsize=(14, 6))
    plt.plot(autocorr)
    plt.title(f'{title} автокорреляция')
    plt.show()

plot_noise(uniform_noise, '')
plot_noise(gaussian_noise, 'Гауссовский белый шум')

# Сумма 100 шумов с равномерным распределением
sum_noise = np.sum([np.random.uniform(-1, 1, N) for _ in range(100)], axis=0)
plot_noise(sum_noise, 'Сумма 100 шумов')
