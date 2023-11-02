""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np
import math

### Ex 1

N = 8
F = np.array([[math.e ** (-2 * np.pi * 1j * k * (m / N)) for m in range(N)] for k in range(N)])
F /= np.sqrt(8)
F_real = [[x.real for x in lin] for lin in F]
F_imag = [[x.imag for x in lin] for lin in F]

for i in range(N):
    fig, axs = plt.subplots(2)
    fig.tight_layout()    
    
    axs[0].stem(F_real[i])
    axs[1].stem(F_imag[i])
    fig.show()
    fig.savefig(f"PS_03_1_{i}.png", format='png')
    fig.savefig(f"PS_03_1_{i}.pdf", format='pdf')
    
conj = np.conjugate(F)
trans = conj.T
unitary = np.dot(F, trans)
identity = np.identity(N)
abs_diff = np.abs(unitary - identity)
epsilon = 1e-5
test = np.all(abs_diff < epsilon)

#%% Ex 2

rx = np.linspace(0, 1, 4000)

def sgn(n):
    return np.cos(2 * np.pi * 5 * n)

def complex_sgn(n, w = 1):
    return sgn(n) * (math.e ** (-2 * np.pi * 1j * n * w))

S = sgn(rx)
CS = complex_sgn(rx)

dist = np.sqrt(CS.real ** 2 + CS.imag ** 2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.tight_layout()   

axs[0].plot(rx, S, color='green')
axs[0].plot(rx[75], S[75], '.', color='red')
axs[1].scatter(CS.real, CS.imag, c=dist, cmap='viridis')
axs[1].plot(CS.real[75], CS.imag[75], '.', color='red')
fig.show()

fig.savefig('PS_03_2_1.png', format='png')
fig.savefig('PS_03_2_1.pdf', format='pdf')

CS_0 = complex_sgn(rx, 1)
CS_1 = complex_sgn(rx, 3)
CS_2 = complex_sgn(rx, 5)
CS_3 = complex_sgn(rx, 7)

dist_0 = np.sqrt(CS_0.real ** 2 + CS_0.imag ** 2)
dist_1 = np.sqrt(CS_1.real ** 2 + CS_1.imag ** 2)
dist_2 = np.sqrt(CS_2.real ** 2 + CS_2.imag ** 2)
dist_3 = np.sqrt(CS_3.real ** 2 + CS_3.imag ** 2)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.tight_layout()
axs[0, 0].scatter(CS_0.real, CS_0.imag, c=dist_0, cmap='viridis')
axs[0, 1].scatter(CS_1.real, CS_1.imag, c=dist_1, cmap='viridis')
axs[1, 0].scatter(CS_2.real, CS_2.imag, c=dist_2, cmap='viridis')
axs[1, 1].scatter(CS_3.real, CS_3.imag, c=dist_3, cmap='viridis')
fig.show()

fig.savefig('PS_03_2_2.png', format='png')
fig.savefig('PS_03_2_2.pdf', format='pdf')

#%% Ex 3

N = 1000

def sgn(t):
    return 2 * np.cos(2 * np.pi * 8 * t) + 4 * np.cos(2 * np.pi * 20 * t) + np.cos(2 * np.pi * 60 * t)
    # return np.cos(2 * np.pi * 5 * t) + 2 * np.cos(2 * np.pi * 20 * t) + 0.5 * np.cos(2 * np.pi * 75 * t)

def get_F(x, w):
    return np.sum(np.array([x[n] * math.e ** ((-2 * np.pi * 1j * w * n) / N) for n in range(N)]))

rx = np.linspace(0, 1, N + 1)
S = sgn(rx)
X = np.array([np.abs(get_F(S, w)) for w in range(1, N + 1)])

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.tight_layout()   

axs[0].plot(rx, S, color='green')
axs[1].stem(X[:100])
fig.show()

fig.savefig('PS_03_3.png', format='png')
fig.savefig('PS_03_3.pdf', format='pdf')
