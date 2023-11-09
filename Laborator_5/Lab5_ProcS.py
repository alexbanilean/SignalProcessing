""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from matplotlib.dates import DateFormatter

### Ex 1
# a) Frecventa de esantionare este de o ora
fs = 1 / 3600

# x = np.genfromtxt("D:\Proiecte\SignalProcessing\Laborator_5\Train.csv", delimiter=',')
x = pd.read_csv("D:\Proiecte\SignalProcessing\Laborator_5\Train.csv")
x['Datetime'] = pd.to_datetime(x['Datetime'])
min_time = x['Datetime'].min()
max_time = x['Datetime'].max()

# b)
print(f"Time interval: {min_time} - {max_time}")

# c)
max_count = x["Count"].max()
print(f'Frecventa maxima: {max_count}')

# d)
x = x.sort_values(by='Datetime')
X = np.fft.fft(x['Count'])
N = len(X)
X = abs(X / N)
X = X[: int(N // 2)]
freqs = fs * np.linspace(0, N / 2, N // 2) / N;

plt.plot(freqs, X, color='green')
plt.title('Modulul Transformatei Fourier')
plt.xlabel('Frecvență (Hz)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.tight_layout()
plt.show()

# e) da, semnalul prezinta o componenta continua
mean_value = np.mean(x['Count'])
x['Count'] = x['Count'] - mean_value
mean_value_check = np.mean(x['Count'])

x = x.sort_values(by='Datetime')
X = np.fft.fft(x['Count'])
N = len(X)
X = abs(X / N)
X = X[: int(N // 2)]
freqs = fs * np.linspace(0, N / 2, N // 2) / N;

plt.plot(freqs, X, color='orange')
plt.title('Modulul Transformatei Fourier')
plt.xlabel('Frecvență (Hz)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.tight_layout()
# plt.show()

# f) 
top_indices = np.argsort(X)[-4:]
top_values = X[top_indices]
top_frequencies = freqs[top_indices]
freqs_to_days = 1 / (60 * 60 * 24 * top_frequencies)

for i in range(4):
    print(f"Frecvența: {top_frequencies[i]} Hz, Amplitudinea: {top_values[i]}, Frecventa in zile: {freqs_to_days[i]}")
    
plt.plot(top_frequencies, top_values, 'o', color='red')
plt.show()

# g)
x = pd.read_csv("D:\Proiecte\SignalProcessing\Laborator_5\Train.csv")
start_date = x['Datetime'].loc[x['Datetime'].dt.day_name() == 'Monday'].min()
filtered_data = x[(x['Datetime'] >= start_date) & (x['Datetime'] < start_date + pd.DateOffset(months=1))]

date_form = DateFormatter("%m-%d")

fig, ax = plt.subplots()

ax.plot(filtered_data['Datetime'], filtered_data['Count'], color='blue')
ax.set_xlabel('Data')
ax.set_ylabel('Trafic')
ax.grid(True)
ax.xaxis.set_major_formatter(date_form)
plt.xticks(rotation=45)
fig.tight_layout()
plt.show()