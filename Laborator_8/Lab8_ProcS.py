from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

### Ex 1

### a)

N = 1000

x = np.arange(1, N + 1)

trend = 0.005 * (x ** 2) + 0.001 * x + 7

season = 160 * np.sin(2 * np.pi * x / 12) + 40 * np.cos(2 * np.pi * x / 36)

noise = 512 * np.random.normal(0, 1, N)

time_series = trend + season + noise

plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.plot(x, trend)
plt.title('Componenta trend')

plt.subplot(4, 1, 2)
plt.plot(x, season)
plt.title('Componenta sezoniera')

plt.subplot(4, 1, 3)
plt.plot(x, noise)
plt.title('Componenta reziduala')

plt.subplot(4, 1, 4)
plt.plot(x, time_series)
plt.title('Serie de timp observata')

plt.tight_layout()
plt.show()

### b)

def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    x_norm = (x - mean) / std
    return x_norm

def autocorr(x):
    n = len(x)
    
    # r = np.zeros(n)
    # for k in range(n):
    #    r[k] = np.correlate(x[:n - k], x[k:])[0]
        
    # x_norm = normalize(x)
    # r = np.correlate(x_norm, x_norm, mode='full')
    # r = r[n - 1:] / np.sum(x_norm ** 2)
    
    r = np.correlate(x, x, mode='full')
    r = r[n - 1:] / np.sum(x ** 2)
    return r

r = autocorr(time_series)

plt.plot(r)
plt.title('Vectorul de autocorelatie')
plt.show()

### c)

def ar_model(x, p):
    n = len(x)
    phi = np.zeros(p)
    for k in range(p):
        phi[k] = np.dot(x[:n - k], x[k:]) / np.dot(x[:n - k], x[:n - k])
    return phi

phi = ar_model(time_series, 6)
print(phi)

y_pred = np.zeros(len(time_series))
x_past = time_series[:6]

for i in range(1, len(time_series)):
  x_past = np.append(x_past[1:], time_series[i - 6])
  y_pred[i] = np.dot(phi, x_past) + noise[i]

plt.plot(time_series, label='Seria de timp originala')
plt.plot(y_pred, label='Predictii')
plt.legend()
plt.tight_layout()
plt.show()
