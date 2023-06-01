import numpy as np
import matplotlib.pyplot as plt
def awgn1(S, SNR):
    n = len(S)               # число отсчетов в сигнале
    Es = np.sum(S**2) / n    # среднеквадратичное значение сигнала
    # SNR = 20 * log10(Es / En)
    En = Es * 10 ** (-SNR / 20)   # среднеквадратичное значение шума
    WGN = np.random.randn(n) * En
    S1 = S + WGN
    return S1
M = 8  # число передаваемых бит
fsig = 1e+2  # частота повторения передаваемой последовательности
N = 1024  # число отсчетов временных и частотных характеристик
i = np.arange(N)  # номера отсчетов для построения характеристик
fs = fsig * N  # частота дискретизации
f = fsig * i  # масштаб по оси частоты
t = 1 / fs * i  # масштаб по оси времени

# bit=[1, 0, 1, 0, 1, 0, 1, 0] # [для демонстрации спектра меандра]
bit = np.round(np.random.rand(M))  # случайный цифровой сигнал из M бит
S = bit[np.ceil((i + 1) * M / N).astype(int) - 1]  # выполняем дискретизацию цифрового сигнала
# на каждый бит приходится N/M дискретных отсчетов
# так как номера элементов в массиве bit начинаются с 1, умножаем на i+1
plt.figure(1)
plt.plot(t, S, '-b')
plt.axis([0, t[-1], -0.5, 1.5])
plt.title('Cигнал (огибающая) до модуляции')
plt.xlabel('t, с')
plt.ylabel('S(t)')

Sp = np.fft.fft(S)  # Cпектр немодулированного сигнала до линии связи
ACH_Sp = np.abs(Sp)  # АЧХ

plt.figure(2)
plt.plot(f, ACH_Sp)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('Спектр сигнала (огибающей) до линии связи')

Knes = 16  # во сколько раз частота несущей больше частоты сигнала
fnes = fsig * Knes  # частота несущей
nes = np.sin(2 * np.pi * f * Knes / fs)  # формирование несущей

S1 = S * nes  # амплитудная модуляция сигнала
plt.figure(3)
plt.plot(t, S1)
plt.axis([0, t[-1], -1.5, 1.5])
plt.title('Модулированный сигнал до линии связи')
plt.xlabel('t, с')
plt.ylabel('S1(t)')

Sp1 = np.fft.fft(S1)  # Cпектр модулированного сигнала до линии связи
ACH_Sp1 = np.abs(Sp1)  # АЧХ

plt.figure(4)
plt.plot(f, ACH_Sp1)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('Спектр модулированного сигнала до линии связи')

l = 1000  # длина линии связи, м
R = 5e-3 + (42e-3) * np.sqrt(f * (1e-6))  # погонное сопротивление
L = 2.7e-7  # погонная индуктивность
G = 20 * f * (1e-15)  # погонная проводимость
C = 48e-12  # погонная емкость

# построение АЧХ и ФЧХ линии связи
w = 2 * np.pi * f  # вектор круговых частот
g1 = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))  # коэффициент распространения волны
K1 = np.exp(-g1 * l)  # комплексная частотная характеристика линии связи
ACH = np.abs(K1)  # АЧХ линии связи
FCH = np.unwrap(np.angle(K1))  # ФЧХ линии связи

plt.figure(5)
plt.subplot(211)  # количество окон по вертикали, по горизонтали, номер окна
plt.semilogx(f, ACH)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|, раз')
plt.title('АЧХ линии связи')

plt.figure(5)
plt.subplot(212)
plt.semilogx(f, FCH)
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('angle(f), радиан')
plt.title('ФЧХ линии связи')

# S1(1:N/2)     - положительные частоты спектра
# S1(N/2+1,N)   - отрицательные частоты спектра
Sp1[N // 2:] = 0  # Зануляем отрицательные частоты в спектре
Sp2 = Sp1 * K1  # Пропускаем сигнал через линию связи

for k in range(1, N // 2):
    Sp2[N // 2 + k] = np.real(Sp2[N // 2 - k]) - 1j * np.imag(Sp2[N // 2 - k])

S2 = np.fft.ifft(Sp2)
plt.figure(6)
plt.plot(t, S2, '-b')
plt.axis([0, t[-1], -1.5, 1.5])
plt.title('Модулированный сигнал после линии связи без шума')
plt.xlabel('t, с')
plt.ylabel('S(t)')

SNR = 2  # соотношение энергий сигнала и шума, в разах
S3 = awgn1(S2,20*np.log(SNR))  # добавление белого гауссовского шума
plt.figure(7)
plt.plot(t, S3, '-b')
plt.axis([0, t[-1], -1.5, 1.5])
plt.title('Модулированный сигнал после линии связи с шумом')
plt.xlabel('t, с')
plt.ylabel('S(t)')

Sp3 = np.fft.fft(S3)
ACH_Sp3 = np.abs(Sp3)  # АЧХ

g = 10  # число ненулевых гармоник огибающей
F1 = np.zeros(N)
F1[Knes - g:Knes + g + 1] = 1  # формируем окно полосового фильтра
F1[N - Knes - g:N - Knes + g + 1] = 1
Sp4 = F1 * Sp3  # пропускаем сигнал через фильтр
plt.figure(8)
plt.stem(f, np.abs(Sp3), 'b')
plt.stem(f, np.abs(Sp4), 'g')
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('Спектр модулированного сигнала до и после ПФ')
plt.legend(['Спектр на входе ПФ', 'Спектр на выходе ПФ'])

S4 = np.fft.ifft(Sp4)  # после фильтрации ПФ переходим во временную область
S5 = np.abs(S4)  # выпрямляем сигнал

Sp5 = np.fft.fft(S5)  # для фильтрации ФНЧ переходим в частотную область

Sp6 = Sp5.copy()
g = 10  # число гармоник, которое пропустит ФНЧ
Sp6[2 + g:N - g] = 0  # ФНЧ: зануляем в спектре гармоники с номером больше g
S6 = np.fft.ifft(Sp6)  # после фильтрации ФНЧ переходим во временную область

plt.figure(9)
plt.stem(f, np.abs(Sp5), 'b')
plt.stem(f, np.abs(Sp6), 'g')
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('Спектр модулированного сигнала до и после ФНЧ')
plt.legend(['Спектр на входе ФНЧ', 'Спектр на выходе ФНЧ'])

plt.figure(10)
plt.plot(t, S4, t, S5, t, S6, t, S)
plt.axis([0, t[-1], -1.5, 1.5])
plt.title('Сигнал после ПФ, выпрямителя и ФНЧ')
plt.xlabel('t, с')
plt.ylabel('S(t)')
plt.legend(['S4 - после ПФ', 'S5 - после выпрямителя', 'S6 - после ФНЧ', 'S - исходный сигнал'])
dt = FCH[1]  # Assuming you want to use the second element of the FCH array
dt = np.round(dt / (2 * np.pi) * N).astype(int)  # переводим из радиан в отсчеты во временной области
print('Задержка в отсчетах сигнала:')
print(dt)

sync = np.zeros(M)
sync[0] = (N / M) / 2 - dt  # первый строб - посередине первого символа + задержка
for i in range(1, M):
    sync[i] = (sync[i-1] + N / M) % N  # сложение в кольце по модулю N
print('Моменты измерения сигнала:')
print(sync)

bit2 = np.zeros(M)
for i in range(M):
    if S6[int(sync[i])] > 0.5:
        bit2[i] = 1
    else:
        bit2[i] = 0
print('Переданный сигнал:')
print(bit)
print('Принятый сигнал:')
print(bit2)
err = np.sum(np.abs(bit - bit2))
print('Число ошибок:')
print(err)
thresholds = np.arange(0, 1.01, 0.01)  # Разные пороговые уровни для исследования
errors1 = np.zeros(len(thresholds))
errors2 = np.zeros(len(thresholds))
total_errors = np.zeros(len(thresholds))
sync = np.array(sync)
sync = sync.astype(int)
for t in range(len(thresholds)):
    thresh = thresholds[t]
    bit2 = np.zeros(M)
    for i in range(M):
        if S6[sync[i]] > thresh:
            bit2[i] = 1
        else:
            bit2[i] = 0
    errors1[t] = np.sum((bit == 0) & (bit2 == 1)) / M  # Ошибка первого рода
    errors2[t] = np.sum((bit == 1) & (bit2 == 0)) / M  # Ошибка второго рода
    total_errors[t] = np.sum(bit != bit2) / M  # Общая ошибка

plt.plot(thresholds, errors1, 'r', thresholds, errors2, 'g', thresholds, total_errors, 'b')
plt.legend(['Ошибка первого рода', 'Ошибка второго рода', 'Общая ошибка'])
plt.xlabel('Порог')
plt.ylabel('Вероятность ошибки')

idx_opt = np.argmin(total_errors)
optimal_threshold = thresholds[idx_opt]
print('Оптимальный порог:')
print(optimal_threshold)
print('Переданный сигнал:')
print(bit)
print('Принятый сигнал:')
print(bit2)
err = np.sum(np.abs(bit - bit2))
print('Число ошибок:')
print(err)
average_threshold = np.mean(S6)
print('Средний порог:')
print(average_threshold)

deviation = np.abs(optimal_threshold - average_threshold)
print('Среднее отклонение от оптимального порога:')
print(deviation)
SNR_values = np.arange(-10, 21, 1) # диапазон значений SNR от -10 до 20 с шагом 1
error_probabilities = np.zeros(len(SNR_values))

for i in range(len(SNR_values)):
    S3 = awgn1(S2, SNR_values[i]) # Добавляем шум с текущим SNR
# [дальнейшая обработка сигнала]
    average_threshold = np.mean(S6)
    bit2 = np.zeros(M)
    for j in range(M):
        if S6[sync[j]] > average_threshold:
            bit2[j] = 1
        else:
            bit2[j] = 0
        err = np.sum(np.abs(bit - bit2))
        error_probabilities[i] = err / M # Считаем вероятность ошибки

plt.figure()
plt.plot(SNR_values, error_probabilities)
plt.title('Зависимость вероятности ошибки от уровня шума')
plt.xlabel('SNR, dB')
plt.ylabel('Вероятность ошибки')
plt.show()