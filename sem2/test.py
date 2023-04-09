import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
fd = 1024
f = 8
ph = 0
A = 1
N = int(fd / f)
t = np.arange(N)

y1 = (A * signal.square(2 * np.pi * t / N+ph, 0.5) + 1) / 2 # меандр (2)
y2 = (A * signal.square(2 * np.pi * t / N+ph, 0.1) + 1) / 2 # меандр (4)

E1_t = np.linalg.norm(y1) ** 2
E1_f = np.linalg.norm(np.abs(np.fft.fft(y1)) ** 2 / len(y1))

E2_t = np.linalg.norm(y2) ** 2
E2_f = np.linalg.norm(np.abs(np.fft.fft(y2)) ** 2 / len(y2))

plt.plot(t, np.abs(np.fft.fft(y1)), 'r', t, np.abs(np.fft.fft(y2)), 'b')
plt.legend(['Меандр, скваж. 2', 'Меандр, скваж. 10'])
plt.show()

print(f"Et исходного сигнала = {E1_t:.0f}")
print(f"Ew исходного сигнала = {E1_f:.0f}")
print(f"Et сигнала после уменьшения ширины = {E2_t:.0f}")
print(f"Ew сигнала после уменьшения ширины = {E2_f:.0f}")
