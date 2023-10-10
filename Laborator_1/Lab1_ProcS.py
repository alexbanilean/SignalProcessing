""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np

### a)

real_axis = np.linspace(0, 0.03, 60)

### b)

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)
 
def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)
 
def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

X = x(real_axis)
Y = y(real_axis)
Z = z(real_axis)

fig_b, axs = plt.subplots(3)
fig_b.tight_layout()

axs[0].plot(real_axis, X)
axs[0].set(xlabel='t', ylabel='x(t)')

axs[1].plot(real_axis, Y)
axs[1].set(xlabel='t', ylabel='y(t)')

axs[2].plot(real_axis, Z)
axs[2].set(xlabel='t', ylabel='z(t)')

fig_b.show()
fig_b.savefig('l1_ex1_b.pdf')

### c)

f = 200

samples = np.linspace(0, 0.1, 20)

X_sample = x(samples)
Y_sample = y(samples)
Z_sample = z(samples)

c_samples = np.linspace(0, 0.1, 200)

X_c_sample = x(c_samples)
Y_c_sample = y(c_samples)
Z_c_sample = z(c_samples)

fig_c, axs = plt.subplots(3)
fig_c.tight_layout()

axs[0].stem(samples, X_sample)
axs[0].plot(c_samples, X_c_sample, color='red')
axs[0].set(xlabel='t', ylabel='x(t)')

axs[1].stem(samples, Y_sample)
axs[1].plot(c_samples, Y_c_sample, color='red')
axs[1].set(xlabel='t', ylabel='y(t)')

axs[2].stem(samples, Z_sample)
axs[2].plot(c_samples, Z_c_sample, color='red')
axs[2].set(xlabel='t', ylabel='z(t)')

fig_c.show()
fig_c.savefig('l1_ex1_c.pdf')
