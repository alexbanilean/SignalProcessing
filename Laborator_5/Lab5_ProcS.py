""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from matplotlib.dates import DateFormatter
from scipy import signal

### Ex 1
# a) Frecventa de esantionare este de o ora
fs = 1 / 3600

# x = np.genfromtxt("D:\Proiecte\SignalProcessing\Laborator_5\Train.csv", delimiter=',')
x = pd.read_csv("D:\Proiecte\SignalProcessing\Laborator_5\Train.csv")
x['Datetime'] = pd.to_datetime(x['Datetime'], format='%d-%m-%Y %H:%M', errors='coerce')\
                .combine_first(pd.to_datetime(x['Datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce'))
min_time, max_time = x['Datetime'].agg(['min', 'max'])

# b)
print(f"Time interval: {min_time} - {max_time}")

# c)
max_count = x["Count"].max()
print(f'Frecventa maxima: {max_count}')

# d)
X = np.fft.fft(x['Count'])
N = len(X)
X = abs(X / N)
X = X[: int(N // 2)]
freqs = fs * np.linspace(0, N / 2, N // 2) / N;

fig, ax = plt.subplots()
ax.plot(freqs, X, color='green')
fig.suptitle('Modulul Transformatei Fourier')
ax.set_xlabel('Frecvență (Hz)')
ax.set_ylabel('Amplitudine')
ax.grid(True)
fig.tight_layout()

fig.savefig('PS_05_d.png', format='png')
fig.savefig('PS_05_d.pdf', format='pdf')

# e) da, semnalul prezinta o componenta continua
mean_value = np.mean(x['Count'])
x['Count'] = x['Count'] - mean_value
mean_value_check = np.mean(x['Count'])

X = np.fft.fft(x['Count'])
N = len(X)
X = abs(X / N)
X = X[: int(N // 2)]
freqs = fs * np.linspace(0, N / 2, N // 2) / N;

fig, ax = plt.subplots()
ax.plot(freqs, X, color='green')
fig.suptitle('Modulul Transformatei Fourier (centrat in 0)')
ax.set_xlabel('Frecvență (Hz)')
ax.set_ylabel('Amplitudine')
ax.grid(True)
fig.tight_layout()

fig.savefig('PS_05_e.png', format='png')
fig.savefig('PS_05_e.pdf', format='pdf')

# f) 
top_indices = np.argsort(X)[-4:]
top_values = X[top_indices]
top_frequencies = freqs[top_indices]
freqs_to_days = 1 / (60 * 60 * 24 * top_frequencies)

for i in range(4):
    print(f"Amplitudinea: {top_values[i]}, Frecventa in zile: {freqs_to_days[i]}")

fig, ax = plt.subplots()
ax.plot(freqs, X, color='green')
ax.plot(top_frequencies, top_values, 'o', color='red')
fig.suptitle('Modulul Transformatei Fourier (centrat in 0)')
ax.set_xlabel('Frecvență (Hz)')
ax.set_ylabel('Amplitudine')
ax.grid(True)
fig.tight_layout()

fig.savefig('PS_05_f.png', format='png')
fig.savefig('PS_05_f.pdf', format='pdf')

# amplitudinile freventelor principale corespund unor perioade de 
# 8 luni, 1 zi, aproximativ 1 an si aproximativ 2 ani

# g)

x['Count'] = x['Count'] + mean_value

# 08-10-2012 -> 07-11-2012
# x_month = x[1056 : 1799]
x_month = x[6768: 7512]

fig, ax = plt.subplots()
ax.plot(x_month['ID'], x_month['Count'])
ax.set_xlabel('Esantioane')
ax.set_ylabel('Trafic')
ax.grid(True)
fig.tight_layout()

fig.savefig('PS_05_g.png', format='png')
fig.savefig('PS_05_g.pdf', format='pdf')

# h)

# Pentru a determina data de start a esantionarii semnalului am putea
# sa ne uitam la un eveniment de referinta(spike) si avand informatii
# despre acest semnal pentru alte intervale cunoscute am putea deduce pe baza
# unor modele de periodicitate(cicluri saptamanale/lunare) a acelui semnal 
# data de start a semnalului curent.

# Factorii care pot influenta acuratetea includ variabilitatea semnalului, 
# calitatea datelor si capacitatea de a identifica corect evenimentul de referinta.

# i)

cutoff = 1 / (60 * 60 * 24) # doar frecventele mai mici de o zi
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist

b, a = signal.butter(4, normal_cutoff, btype='low')
filtered_x_day = signal.lfilter(b, a, x_month['Count'])

cutoff = 1 / (60 * 60 * 24 * 7) # doar frecventele mai mici de o saptamana
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist

b, a = signal.butter(2, normal_cutoff, btype='low')
filtered_x_week = signal.lfilter(b, a, x_month['Count'])

cutoff = 1 / (60 * 60 * 12) # doar frecventele mai mici de 12h
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist

b, a = signal.butter(6, normal_cutoff, btype='low')
filtered_x_p = signal.lfilter(b, a, x_month['Count'])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_month['ID'], x_month['Count'], label='Semnal original')
ax.plot(x_month['ID'], filtered_x_day, label='Zile', color='orange')
ax.plot(x_month['ID'], filtered_x_week, label='Saptamani', color='red')
ax.plot(x_month['ID'], filtered_x_p, label='AM/PM', color='green')
ax.grid(True)
ax.xaxis.set_ticks(np.arange(x_month['ID'][6768], x_month['ID'][7511], 168))
ax.set_xlabel('Esantioane')
ax.set_ylabel('Valoare semnal')
fig.suptitle('Filtrare semnal cu filtru Butterworth lowpass')
fig.legend()
fig.tight_layout()

fig.savefig('PS_05_i.png', format='png')
fig.savefig('PS_05_i.pdf', format='pdf')
