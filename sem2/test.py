import numpy as np
import matplotlib.pyplot as plt

# Этапы преобразования сигнала
# bit - двоичные символы на входе передатчика
# S - исходный сигнал (огибающая)
# S1 - модулированный сигнал на выходе передатчика
# S2 - модулированный сигнал после линии связи
# S3 - модулированный сигнал после линии связи с шумом
# S4 - сигнал после полосового фильтра в приемнике
# S5 - сигнал после выпрямителя
# S6 - сигнал после ФНЧ
# bit2 - двоичные символы на выходе приемника
def calc_error_prob(signal, threshold, original_bits, samples_per_bit):
    num_errors = 0

    for i in range(len(original_bits)):
        start_sample = int((i - 1) * samples_per_bit)
        end_sample = int(i * samples_per_bit)
        average_signal = np.mean(signal[start_sample:end_sample])

        if (average_signal > threshold and original_bits[i] == 0) or (average_signal <= threshold and original_bits[i] == 1):
            num_errors += 1

    prob = num_errors / len(original_bits)
    return prob
# Ввод параметров цифрового сигнала
M = 8  # число передаваемых бит
fsig = 1e+2  # частота повторения передаваемой последовательности
N = 1024  # число отсчетов временных и частотных характеристик
i = np.arange(N)  # номера отсчетов для построения характеристик
fs = fsig * N  # частота дискретизации
f = fsig * i  # масштаб по оси частоты
t = 1 / fs * i  # масштаб по оси времени

# Формирование дискретного цифрового сигнала
# bit = [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0]  # [для демонстрации спектра меандра]
bit = np.round(np.random.rand(M))  # случайный цифровой сигнал из M бит
S = bit[np.ceil((i + 1) * M / N).astype(int) - 1]  # выполняем дискретизацию цифрового сигнала
# на каждый бит приходится N/M дискретных отсчетов
# так как номера элементов в массиве bit начинаеются с 1, умножаем на i+1
plt.figure(1)
plt.plot(t, S, '-b')
plt.axis([0, t[N - 1], -0.5, 1.5])
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

# Ввод параметров и формирование несущей
Knes = 16  # во сколько раз частота несущей больше частоты сигнала
fnes = fsig * Knes  # частота несущей
nes = np.sin(2 * np.pi * f * Knes / fs)  # формирование несущей

# Амплитудная модуляция сигнала
S1 = S * nes  # амплитудная модуляция сигнала
plt.figure(3)
plt.plot(t, S1)
plt.axis([0, t[N - 1], -1.5, 1.5])
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

# Модель линии связи
l = 1000  # длина линии связи, м
R = 5e-3 + (42e-3) * np.sqrt(f * (1e-6))  # погонное сопротивление
L = 2.7e-7  # погонная индуктивность
G = 20 * f * (1e-15)  # погонная проводимость
C = 48e-12  # погонная емкость

# построние АЧХ и ФЧХ линии связи
w = 2 * np.pi * f  # вектор круговых частот
g1 = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))  # коэффициент распространения волны
K1 = np.exp(-g1 * l)  # комплексная частотная характеристика линии связи
ACH = np.abs(K1)  # АЧХ линии связи
FCH = np.unwrap(np.angle(K1))  # ФЧХ линии связи
# функция unwrap убирает скачки фазы, когда значение atan превышает |pi|

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
# ------------ Прохождение сигнала через линию связи --------------------
# S1(1:N/2)     - положительные частоты спектра
# S1(N/2+1:N)   - отрицательные частоты спектра
Sp1[(N//2):] = 0  # Зануляем отрицательные частоты в спектре
Sp2 = Sp1 * K1  # Пропускаем сигнал через линию связи

for k in range(1, N//2):
    Sp2[N//2+k] = np.real(Sp2[N//2-k]) - 1j * np.imag(Sp2[N//2-k])

S2 = np.fft.ifft(Sp2)
plt.figure(6)
plt.plot(t, S2, '-b')
plt.axis([0, t[N-1], -1.5, 1.5])
plt.title('Модулированный сигнал после линии связи без шума')
plt.xlabel('t, с')
plt.ylabel('S(t)')

# ------------ Наложение шума после линии связи -------------------------
SNR = 2  # Соотношение энергий сигнала и шума, в разах
S3 = np.random.normal(0, 1, N)  # Шумовой сигнал
S3 *= np.sqrt(np.sum(np.abs(S2)**2) / np.sum(np.abs(S3)**2)) / SNR  # Масштабируем шум под заданное соотношение
S3 = S3.real.astype(float)
S2 =  S2.real.astype(float)
S3 += S2  # Наложение шума на сигнал
plt.figure(7)
plt.plot(t, S3, '-b')
plt.axis([0, t[N-1], -1.5, 1.5])
plt.title('Модулированный сигнал после линии связи с шумом')
plt.xlabel('t, с')
plt.ylabel('S(t)')

Sp3 = np.fft.fft(S3)
ACH_Sp3 = np.abs(Sp3)  # АЧХ

# ------------ Полосовой фильтр в приемнике -----------------------------
g = 10  # число ненулевых гармоник огибающей
F1 = np.zeros(N)
F1[Knes-g:Knes+g+1] = 1  # формируем окно полосового фильтра
F1[N-Knes-g:N-Knes+g+1] = 1
Sp4 = F1 * Sp3  # пропускаем сигнал через фильтр
plt.figure(8)
plt.stem(f, np.abs(Sp3), 'b')
plt.stem(f, np.abs(Sp4), 'g')
plt.grid(True)
plt.xlabel('f, Гц')
plt.ylabel('АЧХ')
plt.legend(['До фильтра', 'После фильтра'])
plt.title('АЧХ сигнала до и после полосового фильтра')
S4 = np.fft.ifft(Sp4)
plt.figure(9)
plt.plot(t, S4, '-b')
plt.axis([0, t[N-1], -1.5, 1.5])
plt.title('Восстановленный сигнал после полосового фильтра')
plt.xlabel('t, с')
plt.ylabel('S(t)')

# ------------ Демодуляция сигнала ------------------------
S4p = np.zeros(N)
S5=abs(S4)
for k in range(1, N//2):
    S4p[N//2+k] = np.real(S4[N//2-k]) - 1j * np.imag(S4[N//2-k])
# ------------ Фильтр нижней частоты в приемнике ------------------------
Sp5 = np.fft.fft(S5)  # для фильтрации ФНЧ переходим в частотную область
# спектр будет комплексно-сопряженный с постоянной составляющей,
# поэтому занулять его нужно симметрично середине
Sp6 = np.copy(Sp5)
g = 10  # число гармоник, которое пропустит ФНЧ
Sp6[(2 + g):(N - g)] = 0  # ФНЧ: зануляем в спектре гармоники с номером больше g
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
plt.axis([0, t[N-1], -1.5, 1.5])
plt.title('Сигнал после ПФ, выпрямителя и ФНЧ')
plt.xlabel('t, с')
plt.ylabel('S(t)')
plt.legend(['S4 - после ПФ', 'S5 - после выпрямителя', 'S6 - после ФНЧ', 'S - исходный сигнал'])

# ----------- Вычисление задержки сигнала по ФЧХ -------------------------
# определяем задержку в радианах для основной гармоники сигнала,
# чтобы синхронизировать приемник и передатчик
dt = FCH[2]
print('Задержка сигнала в радианах:')
print(dt)
dt = round(dt / (2 * np.pi) * N)  # переводим из радиан в отсчеты во временной области
# (если сдвиг не превышает период сигнала)
# dt = round(rem(dt, 2 * np.pi) / (2 * np.pi) * N)  # если сдвиг превышает период сигнала
# так как во временной области сигнал повторяется с периодом 2 * np.pi,
# то находим только дробную часть периода и переводим ее в отсчеты
print('Задержка в отсчетах сигнала:')
print(dt)

# --------------- Определяем моменты измерения сигнала -------------------
sync = np.zeros(M)
sync[0] = (N / M) / 2 - dt  # первый строб - посередине первого символа + задержка
for i in range(1, M):
    sync[i] = (sync[i - 1] + N / M) % N  # сложение в кольце по модулю N
print('Моменты измерения сигнала:')
print(sync)

# ----------- Пороговый элемент уровню 0.5 -------------------------------
# ВНИМАНИЕ!!! Вместо уровня 0.5 вам нужно использовать другие уровни:
# 1) Оптимальный (задание 2) - определяется по минимальной сумме ошибок 1 и
# 2 рода (сначала нужно построить графики зависимостей ошибок от выбранного
# порога в диапазоне от 0 до 1 с шагом 0.01)
# 2) Средний (задание 3-5) - определяется как среднее арифметическое по
# всем N отсчетам сигнала S6 во временной области
bit2 = np.zeros(M)
for i in range(M):
    if S6[int(sync[i])] > 0.8:
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

thresholds = np.arange(0, 1.01, 0.01)  # Different threshold levels for investigation
errors1 = np.zeros_like(thresholds)
errors2 = np.zeros_like(thresholds)
total_errors = np.zeros_like(thresholds)

for t in range(len(thresholds)):
    thresh = thresholds[t]
    bit2 = np.zeros(M)
    for i in range(M):
        if S6[int(sync[i])] > thresh:
            bit2[i] = 1
        else:
            bit2[i] = 0
    errors1[t] = np.sum((bit == 0) & (bit2 == 1)) / np.sum(bit == 0)  # Error of the first kind
    errors2[t] = np.sum((bit == 1) & (bit2 == 0)) / np.sum(bit == 1)  # Error of the second kind
    total_errors[t] = np.sum(bit != bit2) / M  # Total error

plt.figure()
plt.plot(thresholds, errors1, 'r', thresholds, errors2, 'g', thresholds, total_errors, 'b')
plt.legend(['Ошибка первого рода', 'Ошибка второго рода', 'Общая ошибка'])
plt.xlabel('Порог')
plt.ylabel('Вероятность ошибки')

idx_opt = np.argmin(total_errors)
optimal_threshold = thresholds[idx_opt]
print('Оптимальный порог:')
print(optimal_threshold)
# Вычисление среднего порога
average_threshold = np.mean(S6)
# Вывод значений среднего порога
print('Средний порог:')
print(average_threshold)
# Расчет разницы между оптимальным и средним порогом
deviation = abs(optimal_threshold - average_threshold)
# Вывод среднего отклонения
print('Среднее отклонение от оптимального порога:')
print(deviation)
SNR_values = range(-10, 21, 1)  # диапазон значений SNR от -10 до 20 с шагом 1
error_probabilities = np.zeros(len(SNR_values))

for i in range(len(SNR_values)):
    S3 = np.random.normal(S2, 10 ** (-SNR_values[i] / 20), len(S2))  # Добавляем шум с текущим SNR

    bit2 = np.zeros(M)

    for j in range(M):
        if S3[int(sync[j])] > average_threshold:
            bit2[j] = 1
        else:
            bit2[j] = 0

    err = np.sum(np.abs(bit - bit2))
    error_probabilities[i] = err / M  # Считаем вероятность ошибки

plt.figure()
plt.plot(SNR_values, error_probabilities)
plt.title('Зависимость вероятности ошибки от уровня шума')
plt.xlabel('SNR, dB')
plt.ylabel('Вероятность ошибки')
length_values = np.linspace(1, 100, 100)  # длины линии связи
error_probs = np.zeros(len(length_values))

for i in range(len(length_values)):
    # преобразование длины линии связи в SNR
    # это может быть любое преобразование, соответствующее вашей модели
    SNR = 1 / length_values[i]
    S3 = np.random.normal(S2, 10 ** (20 * np.log10(SNR) / 10), len(S2))  # добавляем шум

    # определение порога по среднему значению принятого сигнала
    threshold = np.mean(S3)

    # расчет вероятности ошибки
    error_probs[i] = calc_error_prob(S3, threshold, bit, N / M)

plt.figure()
plt.plot(length_values, error_probs)
plt.xlabel('Длина линии связи')
plt.ylabel('Вероятность ошибки')
plt.title('Зависимость вероятности ошибки от длины линии связи')
plt.show()