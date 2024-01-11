""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

#%% Exercitiul 1

def get_gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp((-1 / (2 * var)) * ((x - mean) ** 2))

mean = 0
var = 1
x = np.linspace(-10, 10, 1000)

X = get_gaussian(x, mean, var)
data = np.random.normal(mean, np.sqrt(var), 1000)

fig, ax = plt.subplots()
ax.plot(x, X, linewidth=2)
ax.hist(data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
fig.suptitle(f'Distributie gaussiana de medie {mean} si varianta {var}')

ax.grid(True)
fig.tight_layout()

fig.savefig('PS_10_1.png', format='png')
fig.savefig('PS_10_1.pdf', format='pdf')

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

x, y = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
positions = np.column_stack((x.flatten(), y.flatten()))

pdf_values = np.zeros_like(x)
for i in range(positions.shape[0]):
    diff = positions[i] - mean
    exp = np.exp((-1 / 2) * diff.T @ s_inv @ diff)
    pdf_values.flat[i] = 1 / (2 * np.pi * np.sqrt(s_det)) * exp

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)

contour = ax.contour(x, y, pdf_values, colors='red', linewidths=2)

ax.set_xlabel('Axa X')
ax.set_ylabel('Axa Y')
fig.suptitle('Distributie gaussiana 2D')

