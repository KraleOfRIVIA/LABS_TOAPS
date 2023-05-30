import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#--------------- параметры передаваемого сигнала -------------------------
fsig = 1e+3
N = 100
fs = fsig*N
f = np.linspace(0, fsig, N)

#-------------------- параметры линии связи ------------------------------
l = 1000
R = 5e-3 +(42e-3)*np.sqrt(f*1e-6)
L = 2.7e-7
G = 20*f*1e-15
C = 48e-12
w=2*np.pi*f

#----------------- построние АЧХ и ФЧХ линии связи -----------------------
g1 = np.sqrt((R+1j*w*L)*(G+1j*w*C))
K1 = np.exp(-g1*l)
ACH = np.abs(K1)
FCH = np.unwrap(np.angle(K1))

plt.figure(1)
plt.subplot(211)
plt.semilogx(f, ACH, linewidth=2)
plt.grid(True)
plt.xlabel('f, Hz')
plt.ylabel('|K(f)|')
plt.title('ACH of communication line')

plt.subplot(212)
plt.semilogx(f, FCH, linewidth=2)
plt.grid(True)
plt.xlabel('f, Hz')
plt.ylabel('angle(f)')
plt.title('FCH of communication line')

#--------- построение АЧХ и ФЧХ исходного прямоугольного сигнала ---------
A = 1
k = f.size
t = np.arange(k)
y1 = A * np.square(2 * np.pi * t / k)

plt.figure(2)
plt.subplot(121)
plt.plot(t, y1, '-b', linewidth=2)
plt.grid(True)
plt.xlabel('N, number of sample')
plt.ylabel('y1(N)')
plt.title('Original signal in time domain')

S1 = np.fft.fft(y1)
ACH_S1 = np.abs(S1)
FCH_S1 = np.unwrap(np.angle(S1))

plt.figure(3)
plt.subplot(221)
plt.plot(f, ACH_S1)
plt.grid(True)
plt.xlabel('f, Hz')
plt.ylabel('|K(f)|')
plt.title('ACH of original signal')

plt.subplot(222)
plt.plot(f, FCH_S1)
plt.grid(True)
plt.xlabel('f, Hz')
plt.ylabel('angle(f)')
plt.title('FCH of original signal')

#------------- построение АЧХ и ФЧХ сигнала после линии связи ------------
S1[int(N/2+2):] = 0
S2 = S1*K1
ACH_S2 = np.abs(S2)
FCH_S2 = np.unwrap(np.angle(S2))

plt.subplot(223)
plt.stem(f, ACH_S2)
plt.grid(True)
plt.xlabel('f, Hz')
plt.ylabel('|K(f)|')
plt.title('ACH of received signal')

plt.subplot(224)
plt.plot(f, FCH_S2)
plt.grid(True)
plt.xlabel('f, Hz')
plt.ylabel('angle(f)')
plt.title('FCH of received signal')

#---- восстановление сигнала во временной области после линии связи ------
y2 = 2*np.fft.ifft(S2)

plt.figure(2)
plt.subplot(122)
plt.plot(t, np.real(y2), '-r', linewidth=2)
plt.grid(True)
plt.xlabel('N, number of sample')
plt.ylabel('y2(N)')
plt.title('Signal after communication line')
plt.show()
