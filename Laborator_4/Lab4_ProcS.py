""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np
import math
import time

#%%

### Ex 1

def sgn(t):
    return 2 * np.cos(2 * np.pi * 8 * t) + 4 * np.cos(2 * np.pi * 20 * t) + np.cos(2 * np.pi * 60 * t)

def get_F(x, w):
    return np.sum(np.array([x[n] * math.e ** ((-2 * np.pi * 1j * w * n) / N) for n in range(N)]))

N_val = [128, 256, 512, 1024, 2048, 4096, 8192]
my_dft_times = []
numpy_fft_times = []

for N in N_val:
    rx = np.linspace(0, 1, N + 1)
    S = sgn(rx)
    
    start_time = time.time_ns()
    my_dft_result = np.array([np.abs(get_F(S, w)) for w in range(1, N + 1)])
    my_dft_execution_time = time.time_ns() - start_time
    my_dft_times.append(my_dft_execution_time)

    start_time = time.time_ns()
    numpy_fft_result = np.fft.fft(S)
    numpy_fft_execution_time = time.time_ns() - start_time
    numpy_fft_times.append(numpy_fft_execution_time)
 
#%%   
 
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(N_val, my_dft_times, label='my_dft')
ax.plot(N_val, numpy_fft_times, label='numpy.fft')
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Dimensiunea vectorului N')
ax.set_ylabel('Timp de execuție (log scale)')
fig.suptitle('Compararea timpilor de execuție')
fig.legend()
fig.show()

fig.savefig("{}.png".format('PS_04_1'))
fig.savefig("{}.pdf".format('PS_04_1'))

#%% Ex 2

def sgn1(t):
    return np.cos(2 * np.pi * 5 * t)

def sgn2(t):
    return np.cos(2 * np.pi * 3 * t)

def sgn3(t):
    return np.cos(2 * np.pi * 8 * t)

rx = np.linspace(0, 1, 10)
rx2 = np.linspace(0, 1, 100)
S1 = sgn1(rx)
S2 = sgn2(rx)
S3 = sgn3(rx)

fig, axs = plt.subplots(3)
fig.tight_layout()
axs[0].plot(rx, S1, 'o', color="red")
axs[0].plot(rx2, sgn1(rx2), color='blue')
axs[1].plot(rx, S2, 'o', color="red")
axs[1].plot(rx2, sgn2(rx2), color='green')
axs[2].plot(rx, S3, 'o', color="red")
axs[2].plot(rx2, sgn3(rx2), color='orange')
fig.show()

fig.savefig('PS_04_2.png', format='png')
fig.savefig('PS_04_2.pdf', format='pdf')

#%% Ex 3

def sgn1(t):
    return np.cos(2 * np.pi * 5 * t)

def sgn2(t):
    return np.cos(2 * np.pi * 3 * t)

def sgn3(t):
    return np.cos(2 * np.pi * 8 * t)

rx = np.linspace(0, 1, 20)
rx2 = np.linspace(0, 1, 100)
S1 = sgn1(rx)
S2 = sgn2(rx)
S3 = sgn3(rx)

fig, axs = plt.subplots(3)
fig.tight_layout()
axs[0].plot(rx, S1, 'o', color="red")
axs[0].plot(rx2, sgn1(rx2), color='blue')
axs[1].plot(rx, S2, 'o', color="red")
axs[1].plot(rx2, sgn2(rx2), color='green')
axs[2].plot(rx, S3, 'o', color="red")
axs[2].plot(rx2, sgn3(rx2), color='orange')
fig.show()


fig.savefig('PS_04_3.png', format='png')
fig.savefig('PS_04_3.pdf', format='pdf')

#%% Ex 4

"""
Frecventele emise de un contrabas se incadreaza intre 40Hz si 200Hz. Care
este frecventa minima cu care trebuie esantionat semnalul trece-banda
provenit din inregistrarea instrumentului, astfel incat semnalul discretizat
sa contina toate componentele de frecventa pe care instrumentul le poate
produce?
"""
"""
R: Conform teoremei Niquist, frecventa minima de esantionare trebuie sa fie 
400Hz, adica dublul celei mai mari frecvente din semnalul trece-banda (200hz).
"""

#%% Ex 6

import scipy.io.wavfile
import scipy.signal
rate, S = scipy.io.wavfile.read("D:/Proiecte/SignalProcessing/Laborator_4/vowels.wav")
# plt.specgram(S, Fs=rate)

group_size = len(S) // 100
overlap = group_size // 2

groups = [S[i:i + group_size] for i in range(0, len(S), group_size - overlap)]

fft_data = [np.abs(np.fft.fft(group)) for group in groups]

max_amplitudes = [np.max(group) for group in fft_data]
A_max = np.max(max_amplitudes)

fft_matrix_dBFS = 20 * np.log10(np.array(fft_data[:-2]).T / A_max)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(fft_matrix_dBFS, cmap='inferno', origin='lower', aspect='auto')
fig.colorbar(im, label='Amplitudine (dBFS)')
fig.show()

fig.savefig('PS_04_6.png', format='png')
fig.savefig('PS_04_6.pdf', format='pdf')

#%% Ex 7

"""
Puterea unui semnal este Psemnal = 90dB. Se cunoaste raportul semnal 
zgomot, SNRdB = 80dB. Care este puterea zgomotului?
"""
"""
SNR = Psemnal / Pzgomot
SNRdB = 10 log10 SNR
R: 
    SNRdB = 10 log10(Psemnal) - 10 log10(Pzgomot) = 80
    log10(90) - log10(Pzgomot) = 8
    Pzgomot = 10^(log10(90) - 8) = 9e-7
"""
