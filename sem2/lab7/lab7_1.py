import numpy as np
import matplotlib.pyplot as plt

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
S3 = S2 + np.random.normal(0, np.sqrt(np.var(S2) / SNR), N)  # добавление белого гауссовского шума
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
plt.show()