from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

### Ex 1

N = 100
x = np.random.rand(N)

def convolution(x, h):
    M = len(x)
    N = len(h)
    result = np.zeros(M + N - 1)

    for n in range(M + N - 1):
        for k in range(max(0, n - M + 1), min(N, n + 1)):
            result[n] += x[n - k] * h[k]

    return result

x1 = convolution(x, x)
x2 = convolution(x1, x)
x3 = convolution(x2, x)

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

axs[0][0].plot(x)
axs[0][0].set_title('Vector original')

axs[0][1].plot(x1)
axs[0][1].set_title('x ∗ x')

axs[1][0].plot(x2)
axs[1][0].set_title('(x ∗ x) ∗ (x ∗ x)')

axs[1][1].plot(x3)
axs[1][1].set_title('(x ∗ x ∗ x ∗ x) ∗ (x ∗ x ∗ x ∗ x)')

plt.tight_layout()
plt.show()

fig.savefig('PS_06_1.png', format='png')
fig.savefig('PS_06_1.pdf', format='pdf')

#%% Ex 2

N = 5
p_coeff = np.random.randint(-5, 5, size = N + 1)
q_coeff = np.random.randint(-5, 5, size = N + 1)

dp_result = np.convolve(p_coeff, q_coeff)

padded_size = int(2**np.ceil(np.log2(2 * N + 1)))
pp_coeff = np.pad(p_coeff, (0, padded_size - len(p_coeff)))
pq_coeff = np.pad(q_coeff, (0, padded_size - len(q_coeff)))
fftp_result = np.fft.ifft(np.multiply(np.fft.fft(pp_coeff), np.fft.fft(pq_coeff))).real[: 2 * N + 1].round().astype(int)

print("Coeficientii lui p:", p_coeff)
print("Coeficientii lui q:", q_coeff)
print("Convolutie directa:", dp_result)
print("Convolutie cu FFT:", fftp_result)

#%% Ex 3

def rect_window(N):
    return np.ones(N)

def hanning_window(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

def sinewave(t):
    return np.sin(2 * np.pi * 100 * t)

x = np.linspace(0, 1, 200)
S = sinewave(x)

Nw = 200
rS = S * rect_window(Nw)
hS = S * hanning_window(Nw)

fig, axs = plt.subplots(2, figsize=(12, 8))

axs[0].plot(x, rS, color="green")
axs[0].set_title("Sinusoidala trecuta prin fereastra rectangulara")
axs[1].plot(x, hS, color="orange")
axs[1].set_title("Sinusoidala trecuta prin fereastra Hanning")

plt.show()

fig.savefig('PS_06_3.png', format='png')
fig.savefig('PS_06_3.pdf', format='pdf')

#%% Ex 4

import pandas as pd

d = pd.read_csv("D:\Proiecte\SignalProcessing\Laborator_5\Train.csv")
d['Datetime'] = pd.to_datetime(d['Datetime'], format='%d-%m-%Y %H:%M', errors='coerce')\
                .combine_first(pd.to_datetime(d['Datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce'))

# a)
x = d[16464:16536]

# b)

plt.figure(figsize=(12, 12))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan = 2)

ax1.plot(x['Count'])
ax1.set_title('Semnal original')

for i, w in enumerate([5, 9, 13, 17]):
    y = np.convolve(x['Count'], np.ones(w), 'valid') / w
    
    ax = plt.subplot2grid((3, 2), (1 + i // 2, i % 2), colspan = 1)
    
    ax.plot(y)
    ax.set_title(f'Semnal atenuat cu w={w}')
    
plt.tight_layout()
plt.show()

fig.savefig('PS_06_4b.png', format='png')
fig.savefig('PS_06_4b.pdf', format='pdf')    

# c)

f_s = 1 / 3600
f_niq = f_s / 2
f_c = 1 / (3600 * 12) # sau 24
f_norm = f_c / f_niq

print("f_norm =", f_norm)

# d) + e)

fig, axs = plt.subplots(2, figsize=(12, 6))
plt.suptitle("Filtre de ordin 5")

b, a = signal.butter(N = 5, Wn = f_c, btype = 'low', fs = f_s)
filtered_data_butter = signal.lfilter(b, a, x["Count"])

axs[0].plot(x['ID'], x['Count'], label = 'Semnal original')
axs[0].plot(x['ID'], filtered_data_butter, label = 'Semnal filtrat Butterworth', color = "orange")
axs[0].legend()

b, a = signal.cheby1(N = 5, rp = 5, Wn = f_c, btype = 'low', fs = f_s)
filtered_data_cheby = signal.lfilter(b, a, x["Count"])

axs[1].plot(x['ID'], x['Count'], label='Original Signal')
axs[1].plot(x['ID'], filtered_data_cheby, label = 'Semnal filtrat Chebyshev', color = "green")
axs[1].legend()

plt.show()

fig.savefig('PS_06_4e.png', format='png')
fig.savefig('PS_06_4e.pdf', format='pdf')

# f)

fig = plt.figure(figsize=(16, 12))
plt.suptitle("Variatia rp")

for i, rp in enumerate([0.01, 0.1, 1, 10, 100]):
    ax = plt.subplot2grid((5, 2), (2 * i // 2, 2 * i % 2), colspan = 1)
    
    b, a = signal.butter(N = 5, Wn = f_c, btype = 'low', fs = f_s)
    filtered_data_butter = signal.lfilter(b, a, x["Count"])

    ax.plot(x['ID'], x['Count'], label = 'Semnal original')
    ax.plot(x['ID'], filtered_data_butter, label = 'Semnal filtrat Butterworth', color = "orange")
    ax.legend()
    
    ax = plt.subplot2grid((5, 2), ((2 * i + 1) // 2, (2 * i + 1) % 2), colspan = 1)

    b, a = signal.cheby1(N = 5, rp = rp, Wn = f_c, btype = 'low', fs = f_s)
    filtered_data_cheby = signal.lfilter(b, a, x["Count"])

    ax.plot(x['ID'], x['Count'], label='Original Signal')
    ax.plot(x['ID'], filtered_data_cheby, label = f'Semnal filtrat Chebyshev cu rp = {rp}', color = "green")
    ax.legend()


plt.tight_layout()
plt.show()

fig.savefig('PS_06_4f1.png', format='png')
fig.savefig('PS_06_4f1.pdf', format='pdf')

### N = 3

fig, axs = plt.subplots(2, figsize=(12, 6))
plt.suptitle("Filtre de ordin 3")

b, a = signal.butter(N = 3, Wn = f_c, btype = 'low', fs = f_s)
filtered_data_butter = signal.lfilter(b, a, x["Count"])

axs[0].plot(x['ID'], x['Count'], label = 'Semnal original')
axs[0].plot(x['ID'], filtered_data_butter, label = 'Semnal filtrat Butterworth', color = "orange")
axs[0].legend()

b, a = signal.cheby1(N = 3, rp = 0.5, Wn = f_c, btype = 'low', fs = f_s)
filtered_data_cheby = signal.lfilter(b, a, x["Count"])

axs[1].plot(x['ID'], x['Count'], label='Original Signal')
axs[1].plot(x['ID'], filtered_data_cheby, label = 'Semnal filtrat Chebyshev', color = "green")
axs[1].legend()

plt.show()

fig.savefig('PS_06_4f2.png', format='png')
fig.savefig('PS_06_4f2.pdf', format='pdf')

### N = 7

fig, axs = plt.subplots(2, figsize=(12, 6))
plt.suptitle("Filtre de ordin 7")

b, a = signal.butter(N = 7, Wn = f_c, btype = 'low', fs = f_s)
filtered_data_butter = signal.lfilter(b, a, x["Count"])

axs[0].plot(x['ID'], x['Count'], label = 'Semnal original')
axs[0].plot(x['ID'], filtered_data_butter, label = 'Semnal filtrat Butterworth', color = "orange")
axs[0].legend()

b, a = signal.cheby1(N = 7, rp = 0.5, Wn = f_c, btype = 'low', fs = f_s)
filtered_data_cheby = signal.lfilter(b, a, x["Count"])

axs[1].plot(x['ID'], x['Count'], label='Original Signal')
axs[1].plot(x['ID'], filtered_data_cheby, label = 'Semnal filtrat Chebyshev', color = "green")
axs[1].legend()

plt.show()

fig.savefig('PS_06_4f3.png', format='png')
fig.savefig('PS_06_4f3.pdf', format='pdf')
