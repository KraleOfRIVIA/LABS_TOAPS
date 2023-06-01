import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def add_noise(signal, snr):
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise
    return noisy_signal


def encode_name(name):
    encoded_name = [ord(char) for char in name]
    return encoded_name


def decode_name(encoded_name):
    decoded_name = ''.join([chr(char) for char in encoded_name])
    return decoded_name


name = "Artur"
encoded_name = encode_name(name)

snr = 2
threshold = np.arange(0, 1.01, 0.01)

errors_optimal = []
errors_average = []

for t in threshold:
    decoded_name_optimal = ""
    decoded_name_average = ""

    for symbol in encoded_name:
        signal_with_symbol = np.zeros(128)  # Расширяем длину массива для учета всех возможных символов ASCII
        signal_with_symbol[symbol] = 1

        noisy_signal = add_noise(signal_with_symbol, snr)

        noisy_signal_optimal = (noisy_signal > t).astype(int)
        noisy_signal_average = (noisy_signal > 0.5).astype(int)

        decoded_symbol_optimal = np.where(noisy_signal_optimal == 1)[0][0]
        decoded_symbol_average = np.where(noisy_signal_average == 1)[0][0]

        decoded_name_optimal += chr(decoded_symbol_optimal)
        decoded_name_average += chr(decoded_symbol_average)

    errors_optimal.append(sum([1 for i in range(len(name)) if name[i] != decoded_name_optimal[i]]))
    errors_average.append(sum([1 for i in range(len(name)) if name[i] != decoded_name_average[i]]))

plt.plot(threshold, errors_optimal, label="Optimal Threshold")
plt.plot(threshold, errors_average, label="Average Threshold")
plt.xlabel("Threshold")
plt.ylabel("Errors")
plt.legend()
plt.show()

l = 1000
threshold = 0.5
max_noise_level = 0

while True:
    errors = 0

    for symbol in encoded_name:
        signal_with_symbol = np.zeros(l)
        signal_with_symbol[symbol] = 1

        noisy_signal = add_noise(signal_with_symbol, max_noise_level)

        noisy_signal_thresholded = (noisy_signal > threshold).astype(int)

        decoded_symbol = np.where(noisy_signal_thresholded == 1)[0][0]

        if decoded_symbol != symbol:
            errors += 1

    if errors == 0:
        break

    max_noise_level += 0.01

print("Максимальный уровень шума без ошибок:", max_noise_level)
