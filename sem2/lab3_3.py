import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fd = 1024
f = 8
A = 1
k = fd / f
const = 4
t = np.arange(k * const)

y = A * signal.square(2 * np.pi * t / k, 0.125)
y_const = A * signal.square(2 * np.pi * t / (k * const), 0.125)

y1 = A * signal.square(2 * np.pi * t / k, 0.25)
y1_const = A * signal.square(2 * np.pi * t / (k * const), 0.25)

s = np.fft.fft(y)
s_const = np.fft.fft(y_const)
s1 = np.fft.fft(y1)
s1_const = np.fft.fft(y1_const)

AFC = np.abs(s)
AFC_const = np.abs(s_const)
FFCq_const = np.angle(s_const)
FFCq = np.angle(s)

fig, ax = plt.subplots(3, 1, figsize=(6, 9))

ax[0].plot(t / fd, y, 'r')
ax[0].plot(t / fd, y_const, 'g')
ax[0].grid(True)
ax[0].set_title('Square Wave (duty=12.5%)')

ax[1].plot(t * fd / k, AFC_const, 'g')
ax[1].plot(t * fd / k, AFC, 'r')
ax[1].grid(True)
ax[1].set_title('FFT magnitude')

ax[2].plot(t * fd / k, FFCq_const, 'g')
ax[2].plot(t * fd / k, FFCq, 'r')
ax[2].grid(True)
ax[2].set_title('FFT phase')

fig.tight_layout()
plt.show()

AFC = np.abs(s1)
AFC_const = np.abs(s1_const)
FFCq1_const = np.angle(s1_const)
FFCq1 = np.angle(s1)

fig, ax = plt.subplots(3, 1, figsize=(6, 9))

ax[0].plot(t / fd, y1, 'g')
ax[0].plot(t / fd, y1_const, 'r')
ax[0].grid(True)
ax[0].set_title('Square Wave (duty=25%)')

ax[1].plot(t * fd / k, AFC, 'r')
ax[1].plot(t * fd / k, AFC_const, 'g')
ax[1].grid(True)
ax[1].set_title('FFT magnitude')

ax[2].plot(t * fd / k, FFCq1_const, 'g')
ax[2].plot(t * fd / k, FFCq1, 'r')
ax[2].grid(True)
ax[2].set_title('FFT phase')

fig.tight_layout()
plt.show()
