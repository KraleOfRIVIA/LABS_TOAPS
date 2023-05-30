import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import welch

# Параметры сигнала
n_samples = 1000
t = np.linspace(0, 1, n_samples)

# Белый шум с равномерным распределением
uniform_noise = np.random.uniform(-1, 1, size=n_samples)

# Аддитивный белый гауссовский шум
gaussian_noise = np.random.normal(0, 1, size=n_samples)

# Функция для построения графиков
def plot_signals(time, signal, title, xlabel, ylabel):
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Функция для построения спектра
def plot_spectrum(signal, title):
    spectrum = np.abs(fft(signal))
    freq = np.fft.fftfreq(len(signal), 1 / n_samples)
    plt.plot(freq, spectrum)
    plt.title(title)
    plt.xlabel("Частота")
    plt.ylabel("Амплитуда")
    plt.show()

# Функция для построения автокорреляционной функции
def plot_autocorrelation(signal, title):
    autocorr = np.correlate(signal, signal, mode='full')[len(signal)-1:]
    plt.plot(autocorr)
    plt.title(title)
    plt.xlabel("Лаг")
    plt.ylabel("Автокорреляция")
    plt.show()

# Построение графиков для белого шума с равномерным распределением
plot_signals(t, uniform_noise, "Белый шум с равномерным распределением", "Время", "Амплитуда")
plot_spectrum(uniform_noise, "Спектр белого шума с равномерным распределением")
plot_autocorrelation(uniform_noise, "Автокорреляционная функция белого шума с равномерным распределением")

# Построение графиков для аддитивного белого гауссовского шума
plot_signals(t, gaussian_noise, "Аддитивный белый гауссовский шум", "Время", "Амплитуда")
plot_spectrum(gaussian_noise, "Спектр аддитивного белого гауссовского шума")
plot_autocorrelation(gaussian_noise, "Автокорреляционная функция аддитивного белого гауссовского шума")
# Генерация суммы 100 шумов с равномерным распределением
summed_noise = np.zeros(n_samples)
for _ in range(100):
    summed_noise += np.random.uniform(-1, 1, size=n_samples)

# Нормализация суммарного шума
summed_noise /= 100

# Построение графиков для суммарного шума
plot_signals(t, summed_noise, "Сумма 100 шумов с равномерным распределением", "Время", "Амплитуда")
plot_spectrum(summed_noise, "Спектр суммы 100 шумов с равномерным распределением")
plot_autocorrelation(summed_noise, "Автокорреляционная функция суммы 100 шумов с равномерным распределением")