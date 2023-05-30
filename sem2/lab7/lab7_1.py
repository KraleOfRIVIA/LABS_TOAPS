import numpy as np
import matplotlib.pyplot as plt

# Параметры модуляции
carrier_frequency = 100  # Частота несущей сигнала (в герцах)
modulation_index = 0.5  # Индекс модуляции (амплитуда модулирующего сигнала)

# Создание временной оси
duration = 1  # Длительность сигнала (в секундах)
sampling_rate = 1000  # Частота дискретизации (в герцах)
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

# Создание несущего сигнала
carrier_signal = np.sin(2 * np.pi * carrier_frequency * t)

# Создание модулирующего сигнала
modulating_frequency = 10  # Частота модулирующего сигнала (в герцах)
modulating_signal = np.sin(2 * np.pi * modulating_frequency * t)

# Создание амплитудно-модулированного сигнала
am_signal = (1 + modulation_index * modulating_signal) * carrier_signal

# Визуализация сигналов
plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.plot(t, carrier_signal)
plt.title('Carrier Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(t, modulating_signal)
plt.title('Modulating Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)
plt.plot(t, am_signal)
plt.title('AM Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
