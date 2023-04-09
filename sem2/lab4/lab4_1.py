import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
fd = 1024
f = 8
ph = 0
A = 1
N = int(fd / f)
#Дима лох
t = np.arange(N) # временной интервал (сток.)

y_duty2 = (A * signal.square(2 * np.pi * t / N+ph, 0.5) + 1) / 2 # меандр (2)
y_duty4 = (A * signal.square(2 * np.pi * t / N+ph, 0.25) + 1) / 2 # меандр (4)
y_sin = A * np.sin(2 * np.pi * t / N + ph) # синусоида

AS_sin = np.abs(np.fft.fft(y_sin))
AS_duty2 = np.abs(np.fft.fft(y_duty2))
AS_duty4 = np.abs(np.fft.fft(y_duty4))

# Вычисление энергии
Et_sin = np.linalg.norm(y_sin) ** 2
Ew_sin = np.linalg.norm(AS_sin) ** 2 / len(y_sin)

Et_duty2 = np.linalg.norm(y_duty2) ** 2
Ew_duty2 = np.linalg.norm(AS_duty2) ** 2 / len(y_duty2)

Et_duty4 = np.linalg.norm(y_duty4) ** 2
Ew_duty4 = np.linalg.norm(AS_duty4) ** 2 / len(y_duty4)

# Вывод
print(f"fd = {fd}, f = {f}, ph = {ph}, A = {A}, N = {N}")
print(f"Временной интервал от {t[0]} до {t[-1]}")
print(f"Et синусоиды = {Et_sin:.0f}")
print(f"Et пр. импульса, скваж. 2 = {Et_duty2:.0f}")
print(f"Et пр. импульса, скваж. 4 = {Et_duty4:.0f}")
print(f"Ew синусоиды = {Ew_sin:.0f}")
print(f"Ew пр. импульса, скваж. 2 = {Ew_duty2:.0f}")
print(f"Ew пр. импульса, скваж. 4 = {Ew_duty4:.0f}")

# Построение графиков
plt.subplot(3, 1, 1)
plt.plot(t, y_sin)
plt.title("Синусоида")

plt.subplot(3, 1, 2)
plt.plot(t, y_duty2)
plt.title("Прямоугольный импульс, скважность 2")

plt.subplot(3, 1, 3)
plt.plot(t, y_duty4)
plt.title("Прямоугольный импульс, скважность 4")

plt.tight_layout()
plt.show()
