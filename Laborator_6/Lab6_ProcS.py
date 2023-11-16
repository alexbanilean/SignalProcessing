from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

### Ex 1

N = 100
x = np.random.rand(N)

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

for _ in range(3):
    # do it manually here
    x = np.convolve(x, x)

axs[0][0].plot(x[:N])
axs[0][0].set_title('Vector original')

axs[0][1].plot(x[:2 * N - 1])
axs[0][1].set_title('x ∗ x')

axs[1][0].plot(x[:3 * N - 2])
axs[1][0].set_title('(x ∗ x) ∗ (x ∗ x)')

axs[1][1].plot(x[:4 * N - 3])
axs[1][1].set_title('(x ∗ x ∗ x ∗ x) ∗ (x ∗ x ∗ x ∗ x)')

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
