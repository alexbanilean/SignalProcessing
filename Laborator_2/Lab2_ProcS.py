""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np

import scipy.io.wavfile
import scipy.signal
import sounddevice

### Ex 1

def sine_w(t):
    return 2 * np.sin(520 * np.pi * t + np.pi / 6)

def cos_w(t):
    return 2 * np.cos(520 * np.pi * t + np.pi / 6 - np.pi / 2)

real_axis = np.linspace(0, 0.03, 60)
SW = sine_w(real_axis)
CW = cos_w(real_axis)

plt.tight_layout()

plt.plot(real_axis, SW, color='red')
plt.plot(real_axis, CW, color='green')

plt.show()

#%% Ex 2

def b_w(t, p):
    return np.sin(180 * np.pi * t + p)

def get_g(x, z, SNR):
    sx = np.sum(x ** 2)
    sz = np.sum(z ** 2)
    return sx / (SNR * sz)

real_axis = np.linspace(0, 0.03, 600)

Z = np.random.standard_normal(600)

B1 = b_w(real_axis, 0)
B2 = b_w(real_axis, np.pi / 2)
B3 = b_w(real_axis, np.pi / 3)
B4 = b_w(real_axis, np.pi / 6)

SNR = [0.1, 1, 10, 100]

fig_b, axs = plt.subplots(4)
fig_b.tight_layout()

for i, snr in enumerate(SNR):
    gamma1 = get_g(B1, Z, snr)
    X1 = B1 + gamma1 * Z
    axs[i].plot(real_axis, X1, color='cyan')
    
    gamma2 = get_g(B2, Z, snr)
    X2 = B2 + gamma2 * Z
    axs[i].plot(real_axis, X2, color='red')
    
    gamma3 = get_g(B3, Z, snr)
    X3 = B3 + gamma1 * Z
    axs[i].plot(real_axis, X3, color='green')
    
    gamma1 = get_g(B1, Z, snr)
    X1 = B1 + gamma1 * Z
    axs[i].plot(real_axis, X1, color='yellow')

fig_b.show()

#%% Ex 3 

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


fs = 44100
sounddevice.play(signal_a(time_a), fs)
#sounddevice.play(signal_b(time_b_2), fs)
#sounddevice.play(signal_c(time_c), fs)
#sounddevice.play(signal_d(time_d), fs)

rate = int(10e5)
scipy.io.wavfile.write("semnal_ex_2_d.wav", rate, signal_a(time_a))

rate, x = scipy.io.wavfile.read("semnal_ex_2_d.wav")

sounddevice.play(x, fs)

#%% Ex 4

fig, axs = plt.subplots(3)
fig.tight_layout()

time_4 = np.linspace(0, 0.5, 800)

def signal_a(t):
    return np.cos(2 * np.pi * 100 * t)

def signal_c(t):
    return 2 * (60 * t - np.floor(1 / 2 + 60 * t)) 

XA = signal_a(time_4)
XC = signal_c(time_4)
XS = signal_a(time_4) + signal_c(time_4)

axs[0].plot(time_4, XA)
axs[1].plot(time_4, XC)
axs[2].plot(time_4, XS)

fig.show()

#%% Ex 5

def signal_a(t):
    return np.cos(2 * np.pi * 700 * t)

def signal_b(t):
    return np.cos(2 * np.pi * 200 * t)

time_5 = np.linspace(0, 10, 88200)

XA = signal_a(time_5)
XB = signal_b(time_5)
XS = np.append(XA, XB)

sounddevice.play(XS, 44100)

#%% Ex 6

def signal_6(t, fs):
    return np.sin(2 * np.pi * fs * t)

fs = 100
time_6 = np.linspace(0, 0.05, fs)

XA = signal_6(time_6, fs / 2)
XB = signal_6(time_6, fs / 4)
XC = signal_6(time_6, 0)

plt.tight_layout()

plt.plot(time_6, XA, color='red')
plt.plot(time_6, XB, color='green')
plt.plot(time_6, XC, color='cyan')

plt.show()

#%% Ex 7

def signal_7(t):
    return np.sin(2 * np.pi * 300 * t)

time_7 = np.linspace(0, 0.025, 25)

XA = signal_7(time_7)
XB = XA[::4]
XC = XA[1::4]

fig, axs = plt.subplots(2)
fig.tight_layout()

axs[0].plot(time_7, XA, color='red')
axs[0].plot(time_7[::4], XB, 'o', color='green')
axs[0].plot(time_7[1::4], XC, 'o', color='cyan')

axs[1].plot(time_7[::4], XB, color='green')
axs[1].plot(time_7[1::4], XC, color='cyan')

fig.show()

#%% Ex 8

alpha = np.linspace(-np.pi / 2, np.pi / 2, 1000)
sin_alpha = np.sin(alpha)
pade_alpha = (alpha - 7 * alpha**3 / 60) / (1 + alpha**2 / 20)

sin_error = np.abs(alpha - sin_alpha)
pade_error = np.abs(alpha - pade_alpha)

fig, axs = plt.subplots(2)
fig.tight_layout()

axs[0].plot(alpha, alpha, color='green')
axs[0].plot(alpha, sin_alpha, color='blue', label = 'sin alpha')
axs[0].plot(alpha, pade_alpha, color='yellow', label = 'pade alpha')
axs[0].set(xlabel='alpha', ylabel='value')

axs[1].plot(alpha, sin_error, color='red')
axs[1].plot(alpha, pade_error, color='orange')
axs[1].set(xlabel='alpha', ylabel='error value')

fig.show()

