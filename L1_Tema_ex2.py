""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(4)
fig.tight_layout()

### a)

def signal_a(t):
    return np.cos(2 * np.pi * 400 * t)

time_a = np.linspace(0, 0.05, 1600)

axs[0].plot(time_a, signal_a(time_a))
axs[0].set_title('Semnal sinusoidal cu f0=400Hz si samples=1600')

### b)

def signal_b(t):
    return np.cos(2 * np.pi * 800 * t)

time_b = np.linspace(0, 3, 20)
time_b_2 = np.linspace(0, 3, 400) 

axs[1].stem(time_b, signal_b(time_b))
axs[1].plot(time_b_2, signal_b(time_b_2), color='orange')
axs[1].set_title('Semnal sinusoidal cu f0=800Hz si t=0:3')

### c)

def signal_c(t):
    return 2 * (240 * t - np.floor(1 / 2 + 240 * t)) 

time_c = np.linspace(0, 0.05, 240)

axs[2].plot(time_c, signal_c(time_c))
axs[2].set_title('Semnal sawtooth cu f0=240Hz')

### d)

def signal_d(t):
    return np.sign(np.cos(2 * np.pi * 300 * t))

time_d = np.linspace(0, 0.05, 1200)

axs[3].plot(time_d, signal_d(time_d))
axs[3].set_title('Semnal square cu f0=300Hz')

fig.show()

### e)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout()

signal_e = np.random.rand(128, 128)
axs[0].imshow(signal_e)
axs[0].set_title('Semnal 2D aleator')

### f)

x, y = np.meshgrid(np.arange(128), np.arange(128))
signal_f = np.sin(2 * np.pi * x + np.cos(y)) + np.cos(2 * np.pi * x)

axs[1].imshow(signal_f)
axs[1].set_title('Semnal 2D la alegere')

fig.show()
