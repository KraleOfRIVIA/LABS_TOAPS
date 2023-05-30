import numpy as np
import matplotlib.pyplot as plt

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
plt.hold(True)
plt.stem(f, np.abs(Sp6), 'g')
plt.grid(True)
plt.hold(False)
plt.xlabel('f, Гц')
plt.ylabel('|K(f)|')
plt.title('Спектр модулированного сигнала до и после ФНЧ')
plt.legend(['Спектр на входе ФНЧ', 'Спектр на выходе ФНЧ'])

plt.figure(10)
plt.plot(t, S4, t, S5, t, S6, t, S)
plt.axis([0, t[N], -1.5, 1.5])
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
# -------------- Определение порога для бинарной классификации --------------
# В данном примере применяется пороговый элемент уровню 0.5.
# Однако, вам следует использовать подходящий порог в соответствии с задачей.

threshold = 0.5

bit2 = np.zeros(M)
for i in range(M):
    if S6[int(sync[i])] > threshold:
        bit2[i] = 1
    else:
        bit2[i] = 0

# Ваши дальнейшие операции с переменной 'bit2' здесь...
error_probabilities = []

for SNR_value in SNR_values:
    S3 = np.random.normal(S2, 10 ** (-SNR_value / 20), size=S2.shape)  # Adding noise with current SNR

    bit2 = np.zeros(M)

    for j in range(M):
        if S3[int(sync[j])] > average_threshold:
            bit2[j] = 1
        else:
            bit2[j] = 0

    err = np.sum(np.abs(bit - bit2))
    error_probabilities.append(err / M)
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
plt.show()
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