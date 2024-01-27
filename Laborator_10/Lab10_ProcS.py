""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

#%% Exercitiul 1

def get_gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp((-1 / (2 * var)) * ((x - mean) ** 2))

mean = 0
var = 4
x = np.linspace(-10, 10, 1000)

X = get_gaussian(x, mean, var)
data = np.random.normal(mean, np.sqrt(var), 1000)

fig, ax = plt.subplots()
ax.plot(x, X, linewidth=2)
ax.hist(data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
fig.suptitle(f'Distributie gaussiana de medie {mean} si varianta {var}')

ax.grid(True)
fig.tight_layout()

fig.savefig('PS_10_1_0.png', format='png')
fig.savefig('PS_10_1_0.pdf', format='pdf')

#%%

def get_2d_gaussian_samples(x, mean, sigma):
    eigenvalues, eigenvectors = LA.eig(sigma)
    u = np.vstack([e for e in eigenvectors])
    
    return u @ np.diag([np.sqrt(e) for e in eigenvalues]) @ x + mean

mean = np.array([0, 0])
sigma = np.array([[1, 3/5], [3/5, 2]])

# get n from the 2d normal distribution of 0 mean and I2 variance
n = np.random.multivariate_normal(mean, np.array([[1, 0], [0, 1]]), 1000)
X = np.array([get_2d_gaussian_samples(x, mean, sigma) for x in n])

s_det = LA.det(sigma)
s_inv = LA.inv(sigma)

x, y = np.meshgrid(np.linspace(-5, 5, 1000), np.linspace(-5, 5, 1000))
positions = np.dstack((x, y))
diff = positions - mean
exp = np.exp((-1 / 2) * np.sum(diff @ s_inv * diff, axis=2))
pdf_values = (1 / (2 * np.pi * np.sqrt(s_det))) * exp

fig, ax = plt.subplots(3, figsize=(12, 12))

ax[0].scatter(X[:, 0], X[:, 1], c='black', alpha=0.5)
contour = ax[0].contour(x, y, pdf_values, levels=[np.min(pdf_values) + 0.01], colors='lime', linewidths=2)

ax[0].grid(True)
ax[0].set_xlabel('Axa X')
ax[0].set_ylabel('Axa Y')
ax[0].set_title('Distributie gaussiana 2D')

ax[1].hist(X[:, 0], bins=40, alpha=0.7, density=True)
ax[1].plot(np.linspace(-5, 5, 1000), get_gaussian(np.linspace(-5, 5, 1000), mean[0], sigma[0, 0]), color='blue', linewidth=2)
ax[1].set_title('Marginal Distribution along X-axis')

ax[2].hist(X[:, 1], bins=40, alpha=0.7, density=True)
ax[2].plot(np.linspace(-5, 5, 1000), get_gaussian(np.linspace(-5, 5, 1000), mean[1], sigma[1, 1]), color='red', linewidth=2)
ax[2].set_title('Marginal Distribution along Y-axis')

fig.tight_layout()
fig.savefig('PS_10_1_1.png', format='png')
fig.savefig('PS_10_1_1.pdf', format='pdf')


#%% Exercitiul 2

def get_2d_gaussian_samples(mean, sigma):
    n = np.random.normal(size=len(mean))
    eigenvalues, eigenvectors = LA.eig(sigma)
    u = np.vstack([e for e in eigenvectors])
    
    return u @ np.diag([np.sqrt(e) for e in eigenvalues]) @ n + mean

fig, ax = plt.subplots(5, figsize=(8, 20))

# brownian PG

cnt_r = 150
r = np.linspace(0, 4, cnt_r)

mean = np.zeros(cnt_r)
sigma = np.zeros((cnt_r, cnt_r))

for i in range(cnt_r):
    for j in range(cnt_r):
        sigma[i, j] = min(i, j)
        
Z = get_2d_gaussian_samples(mean, sigma)
ax[0].plot(Z, marker='.')
ax[0].set_xticks(np.arange(0, cnt_r, step=20))
ax[0].set_xticklabels([f'{x:.2f}' for x in r[::20]])
ax[0].set_title('Brownian Gaussian Process')
ax[0].grid(True)

# square PG

cnt_r = 150
r = np.linspace(-1, 1, cnt_r)
alpha = 100

mean = np.zeros(cnt_r)
sigma = np.zeros((cnt_r, cnt_r))

for i in range(cnt_r):
    for j in range(cnt_r):
        sigma[i, j] = np.exp(-alpha * np.sum((r[i] - r[j]) ** 2))
        
Z = get_2d_gaussian_samples(mean, sigma)
ax[1].plot(Z, marker='.')
ax[1].set_xticks(np.arange(0, cnt_r, step=20))
ax[1].set_xticklabels([f'{x:.2f}' for x in r[::20]])
ax[1].set_title('Squared exponential Gaussian Process')
ax[1].grid(True)

# ornstein-uhlenbeck PG

cnt_r = 150
r = np.linspace(0, 4, cnt_r)
alpha = 2

mean = np.zeros(cnt_r)
sigma = np.zeros((cnt_r, cnt_r))

for i in range(cnt_r):
    for j in range(cnt_r):
        sigma[i, j] = np.exp(-alpha * np.abs(r[i] - r[j]))
        
Z = get_2d_gaussian_samples(mean, sigma)
ax[2].plot(Z, marker='.')
ax[2].set_xticks(np.arange(0, cnt_r, step=20))
ax[2].set_xticklabels([f'{x:.2f}' for x in r[::20]])
ax[2].set_title('Ornstein-Uhlenbeck Gaussian Process')
ax[2].grid(True)

# periodical PG

cnt_r = 150
r = np.linspace(-2, 2, cnt_r)
alpha = 4
beta = 6

mean = np.zeros(cnt_r)
sigma = np.zeros((cnt_r, cnt_r))

for i in range(cnt_r):
    for j in range(cnt_r):
        sigma[i, j] = np.exp(-alpha * np.sin(beta * np.pi * (r[i] - r[j])) ** 2)
        
Z = get_2d_gaussian_samples(mean, sigma)
ax[3].plot(Z, marker='.')
ax[3].set_xticks(np.arange(0, cnt_r, step=20))
ax[3].set_xticklabels([f'{x:.2f}' for x in r[::20]])
ax[3].set_title('Periodical Gaussian Process')
ax[3].grid(True)

# symmetrical PG

cnt_r = 150
r = np.linspace(-4, 4, cnt_r)
alpha = 20

mean = np.zeros(cnt_r)
sigma = np.zeros((cnt_r, cnt_r))

for i in range(cnt_r):
    for j in range(cnt_r):
        sigma[i, j] = np.exp(-alpha * np.minimum(np.abs(r[i] - r[j]), np.abs(r[i] + r[j])) ** 2)
        
Z = get_2d_gaussian_samples(mean, sigma)
ax[4].plot(Z, marker='.')
ax[4].set_xticks(np.arange(0, cnt_r, step=20))
ax[4].set_xticklabels([f'{x:.2f}' for x in r[::20]])
ax[4].set_title('Symmetrical Gaussian Process')
ax[4].grid(True)

fig.tight_layout()
fig.savefig('PS_10_2.png', format='png')
fig.savefig('PS_10_2.pdf', format='pdf')

#%% Exercitiul 3

import pandas as pd
from sklearn.datasets import fetch_openml

co2 = fetch_openml(data_id=41187, as_frame=True, parser='liac-arff')

#%%
df = co2.frame
monthly_avg = df.groupby(['year', 'month'])['co2'].mean()
monthly_avg.index = monthly_avg.index.map(lambda x: f'{x[0]}-{x[1]:02f}')

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(range(len(monthly_avg)), monthly_avg.values, linestyle='-', color='b')
ax.set_title('Media lunara a concentratiei de CO2')
ax.set_xlabel('Index')
ax.set_ylabel('Concentratie CO2')
ax.grid(True)

fig.tight_layout()
fig.savefig('PS_10_3_1.png', format='png')
fig.savefig('PS_10_3_1.pdf', format='pdf')

#%%
q = 12

moving_average = np.convolve(monthly_avg, np.ones(q) / q, mode='valid')
detrended_series = monthly_avg.iloc[(q - 1) :] - moving_average

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_avg.values, label='Original Time Series', linestyle='-')
ax.plot(range(q - 1, len(moving_average) + q - 1), moving_average, label='Trend Component', linestyle='--', color='red')
ax.plot(range(q - 1, len(detrended_series) + q - 1), detrended_series.values, label='Detrended Series', linestyle='-', color='green')
ax.set_title('Original Time Series, Trend Component, and Detrended Series')
ax.set_xlabel('Index')
ax.set_ylabel('Concentratie CO2')
ax.legend()
ax.grid(True)
ax.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
ax.yaxis.grid(which='major', linestyle='-', linewidth='0.5', color='gray')

fig.tight_layout()
fig.savefig('PS_10_3_2.png', format='png')
fig.savefig('PS_10_3_2.pdf', format='pdf')

#%%

len_pred = 5 * 12 

X_train = np.array(range(len(detrended_series.index) - len_pred + 1))
Y_train = np.array(detrended_series.values[:-len_pred + 1])
X_test = np.array(range(len(detrended_series.index) - len_pred, len(detrended_series)))

def get_2d_gaussian_samples(mean, sigma):
    n = np.random.normal(size=len(mean))
    eigenvalues, eigenvectors = LA.eigh(sigma)
    u = np.vstack([e for e in eigenvectors])
    
    return u @ np.diag([np.sqrt(max(0, e.real)) for e in eigenvalues]) @ n + mean

def get_sigma(r1, r2):
    l2 = 1
    sf2 = 2
    sigma = np.zeros((len(r1), len(r2)))

    for i in range(len(r1)):
        for j in range(len(r2)):
            sigma[i, j] = sf2 * np.exp((-1 / (2 * l2)) * np.abs(r1[i] - r2[j]) ** 2)
    
    return sigma
            
sn2 = 0.2

sigma_test_test = get_sigma(X_test, X_test)
sigma_test_train = get_sigma(X_test, X_train)
sigma_train_test = get_sigma(X_train, X_test)
sigma_train_train = get_sigma(X_train, X_train)

sigma_train_train += np.eye(len(X_train)) * sn2

m = np.mean(detrended_series.values) + sigma_test_train @ LA.pinv(sigma_train_train) @ (Y_train - np.mean(Y_train))
D = sigma_test_test - sigma_test_train @ LA.pinv(sigma_train_train) @ sigma_train_test

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(X_train, Y_train, color='red', label='Measurements')
#ax.plot(X_test, detrended_series.values[-len_pred:], color='green', label='Actual Values (Test)')
#ax.scatter(X_test, detrended_series.values[-len_pred:], color='red')
ax.plot(X_test, m, color='orange', label="Predictions mean")

cnt_pred = 1
Y_predicted = np.array([get_2d_gaussian_samples(m, D) for _ in range(cnt_pred)]).T

#for i in range(cnt_pred):
    #ax.scatter(X_test, Y_predicted[:, i])

for i in range(cnt_pred):
    ax.plot(X_test, Y_predicted[:, i], linestyle='dashed', color='blue', label='Predictions')

ax.legend()
ax.set_title('Regresie cu Proces Gaussian')
ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.tight_layout()
fig.savefig('PS_10_3_3.png', format='png')
fig.savefig('PS_10_3_3.pdf', format='pdf')
