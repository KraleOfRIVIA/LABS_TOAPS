import numpy as np
import matplotlib.pyplot as plt

fsig = 1e+4  # частота первой гармоники сигнала
N = 100  # число отсчетов характеристики + 1 (нулевая гармоника)
fs = fsig * N  # частота дискретизации
f = np.arange(0, fs + fsig, fsig)  # диапазон частот для АЧХ сигнала

l = 1000  # длина линии связи, м
R = 5e-3 + (42e-3) * np.sqrt(f * 1e-6)  # погонное сопротивление
L = 2.7e-7  # погонная индуктивность
G = 20 * f * 1e-15  # погонная проводимость
C = 48e-12  # погонная емкость

# построение АЧХ и ФЧХ линии связи
w = 2 * np.pi * f  # вектор круговых частот
g1 = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))  # коэффициент распространения волны
K1 = np.exp(-g1 * l)  # комплексная частотная характеристика линии связи

ACH = np.abs(K1)  # АЧХ линии связи
FCH = np.unwrap(np.angle(K1))  # ФЧХ линии связи

# график АЧХ
plt.figure()
plt.subplot(211)
plt.semilogx(f, ACH)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('АЧХ линии связи')

# график ФЧХ
plt.subplot(212)
plt.semilogx(f, FCH)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('angle(f)')
plt.title('ФЧХ линии связи')

plt.show()
