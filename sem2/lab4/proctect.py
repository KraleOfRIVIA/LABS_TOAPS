import numpy as np # необходима для массивов и мат. операций
import librosa # неоходима для получения массива значений из аудиофайла и его частоты дискретизации
import matplotlib.pyplot as plt # для графиков
audio_file = 'dictaphone.audio-_2_.wav' # загружаем аудиофайл
signal, sample_rate = librosa.load(audio_file, sr=None)
print("Частота дискретизации:",sample_rate)
fig, ax = plt.subplots()
ax.plot(signal)
time_energy = np.sum(signal ** 2)
print("Энергия во временной области:",time_energy)
spectrum = np.fft.fft(signal)
freq_energy = np.sum(np.abs(spectrum) ** 2) / len(spectrum)
print("Энергия в частотной области:",freq_energy)
freqs = np.fft.fftfreq(len(spectrum), 1 / sample_rate)
fundamental_freq = np.abs(freqs[np.argmax(np.abs(spectrum))])
max_freq = np.max(freqs)
bandwidth = max_freq - fundamental_freq
print("Ширина спектра:",bandwidth)
poisson_theorem = np.isclose(time_energy, freq_energy, rtol=1e-5, atol=1e-8)
print("Проверка на теорему Пуассона:",poisson_theorem)
frequencies = np.fft.fftfreq(len(spectrum), 1 / sample_rate)
# Амплитудно-частотная характеристика (АЧХ)
amplitude_spectrum = np.abs(spectrum)

# Фазово-частотная характеристика (ФЧХ)
phase_spectrum = np.angle(spectrum)

# Вычисляем среднюю частоту
mean_freq = np.sum(frequencies * np.abs(spectrum)) / np.sum(np.abs(spectrum))

# Вычисляем стандартное отклонение частоты
std_freq = np.sqrt(np.sum((frequencies - mean_freq)**2 * np.abs(spectrum)) / np.sum(np.abs(spectrum)))

# Вычисляем эффективную частоту пропускания
effective_bandwidth = 2 * std_freq
print("Эффективная частота пропускания:", effective_bandwidth)


# Построение графиков
plt.figure(figsize=(12, 5))

# График АЧХ
plt.subplot(1, 2, 1)
plt.plot(frequencies, amplitude_spectrum)
plt.title('Амплитудно-частотная характеристика (АЧХ)')
plt.xlabel('Частота (Hz)')
plt.ylabel('Амплитуда')

# График ФЧХ
plt.subplot(1, 2, 2)
plt.plot(frequencies, phase_spectrum)
plt.title('Фазово-частотная характеристика (ФЧХ)')
plt.xlabel('Частота (Hz)')
plt.ylabel('Фаза (радианы)')

plt.tight_layout()
plt.show()