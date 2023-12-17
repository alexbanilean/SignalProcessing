""" Alexandru Banilean, grupa 352 """

import matplotlib.pyplot as plt
import numpy as np

### Ex 1

### a)

N = 1000

x = np.arange(1, N + 1)

trend = 0.005 * (x ** 2) + 0.001 * x + 7

season = 160 * np.sin(2 * np.pi * x / 12) + 40 * np.cos(2 * np.pi * x / 36)

noise = (N / 16) * np.random.normal(0, 1, N)

time_series = trend + season + noise

fig, axs = plt.subplots(4, figsize=(12, 6))

axs[0].plot(x, trend)
axs[0].set_title('Componenta trend')

axs[1].plot(x, season)
axs[1].set_title('Componenta sezoniera')

axs[2].plot(x, noise)
axs[2].set_title('Componenta reziduala')

axs[3].plot(x, time_series)
axs[3].set_title('Serie de timp observata')

plt.tight_layout()
plt.show()

fig.savefig('PS_09_1.png', format='png')
fig.savefig('PS_09_1.pdf', format='pdf')

### b)

def med_exp(x, alpha):
    s = np.zeros_like(x)
    s[0] = x[0]
    
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1 - alpha) * s[t - 1]
        
    return s

alpha = 0.35
nts = med_exp(time_series, alpha)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x, time_series, label="Seria de timp originala")
ax.plot(x, nts, color='orange', label="Seria de timp rezultata")
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Serie de timp rezultata din medierea exponentiala alpha={alpha:.3f}')
plt.show()

fig.savefig('PS_09_2.png', format='png')
fig.savefig('PS_09_2.pdf', format='pdf')

lmt = 100

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x[:lmt], time_series[:lmt], label='Serie de timp originala')
ax.plot(x[:lmt], nts[:lmt], color='orange', label="Seria de timp rezultata")
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Serie de timp rezultata din medierea exponentiala alpha={alpha:.3f}')
plt.show()

fig.savefig('PS_09_2_1.png', format='png')
fig.savefig('PS_09_2_1.pdf', format='pdf')


def loss_function(x, alpha):
    s = med_exp(x, alpha)
    return np.sum((s[:-1] - x[1:]) ** 2)

def get_best_alpha(x, alpha_values):
    best_loss = np.Infinity
    best_alpha = None
    
    for alpha in alpha_values:
        loss = loss_function(x, alpha)
        
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            
    # print(best_alpha, best_loss)
            
    return best_alpha
    
    
alpha_values = np.linspace(0, 1, 1000)
best_alpha = get_best_alpha(time_series, alpha_values)

nts = med_exp(time_series, best_alpha)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x, time_series, label="Seria de timp originala")
ax.plot(x, nts, color='orange', label="Seria de timp rezultata")
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Serie de timp rezultata din medierea exponentiala cu alpha={best_alpha:.3f} optim')
plt.show()

fig.savefig('PS_09_2_2.png', format='png')
fig.savefig('PS_09_2_2.pdf', format='pdf')

lmt = 100

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x[:lmt], time_series[:lmt], label='Serie de timp originala')
ax.plot(x[:lmt], nts[:lmt], color='orange', label="Seria de timp rezultata")
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Serie de timp rezultata din medierea exponentiala cu alpha={best_alpha:.3f} optim')
plt.show()

fig.savefig('PS_09_2_3.png', format='png')
fig.savefig('PS_09_2_3.pdf', format='pdf')

### c)

q = 5
m = N - q

miu = np.mean(time_series)

def estimate_ma_coefficients(ts, q, m, miu):
    eps = np.random.normal(size=len(ts))
    y = ts[q:q+m]

    Y = np.column_stack([eps[i - q : i + 1][::-1].tolist() + [miu] for i in range(q, q + m)])
    Y = Y.T

    x = np.linalg.lstsq(Y[:, 1:-1], y, rcond=None)[0]
    
    x_estimated = np.concatenate(([1], x, [1]))

    return x_estimated
    
def predict_ma_model(ts, q, m, x, miu):
    n = len(ts)
    eps = np.random.normal(size=n)
    predictions = np.zeros(n)
    
    for i in range(q, n):
        x_pred = np.array(eps[i - q : i + 1][::-1].tolist() + [miu])
        
        predictions[i] = np.dot(x, x_pred)
    
    return predictions
    
x_estimated = estimate_ma_coefficients(time_series, q, m, miu)
predictions = predict_ma_model(time_series, q, m, x_estimated, miu)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x, time_series, label='Serie de timp originala')
ax.plot(x, predictions, label=f'Predictii MA(q={q})', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Model MA(q={q}) și predictii')
plt.show()

fig.savefig('PS_09_3.png', format='png')
fig.savefig('PS_09_3.pdf', format='pdf')

lmt = 100

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x[:lmt], time_series[:lmt], label='Serie de timp originala')
ax.plot(x[:lmt], predictions[:lmt], label=f'Predictii MA(q={q})', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Model MA(q={q}) si predictii')
plt.show()

fig.savefig('PS_09_3_1.png', format='png')
fig.savefig('PS_09_3_1.pdf', format='pdf')

### d)

p = 12
q = 5
m = N - max(p, q)
miu = np.mean(time_series)

def estimate_arma_coefficients(ts, p, q, m, miu):
    y = ts[p:p+m]
    
    Y = np.column_stack([ts[i - p : i][::-1] for i in range(p, p + m)])
    Y = Y.T
    
    x_AR = np.linalg.lstsq(Y, y, rcond=None)[0]
    
    eps = np.random.normal(size=len(ts))
    y = ts[q:q+m]

    Y = np.column_stack([eps[i - q : i + 1][::-1].tolist() + [miu] for i in range(q, q + m)])
    Y = Y.T

    x = np.linalg.lstsq(Y[:, 1:-1], y, rcond=None)[0]
    
    x_MA = np.concatenate(([1], x, [1]))

    return x_AR, x_MA

def predict_arma_model(ts, p, q, m, x_AR, x_MA, miu):
    n = len(ts)
    eps = np.random.normal(size=n)
    predictions = np.zeros(n)
    
    for i in range(max(p, q), n):
        x_pred_AR = np.array(ts[i - p : i][::-1])
        x_pred_MA = np.array(eps[i - q : i + 1][::-1].tolist() + [miu])
        
        predictions[i] = np.dot(x_AR, x_pred_AR) + np.dot(x_MA, x_pred_MA)
    
    return predictions

x_AR, x_MA = estimate_arma_coefficients(time_series, p, q, m, miu)
predictions = predict_arma_model(time_series, p, q, m, x_AR, x_MA, miu)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x, time_series, label='Serie de timp originala')
ax.plot(x, predictions, label=f'Predictii ARMA(p={p},q={q})', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Model ARMA(p={p},q={q}) si predictii')
plt.show()

fig.savefig('PS_09_4.png', format='png')
fig.savefig('PS_09_4.pdf', format='pdf')

lmt = 100

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x[:lmt], time_series[:lmt], label='Serie de timp originala')
ax.plot(x[:lmt], predictions[:lmt], label=f'Predictii ARMA(p={p},q={q})', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Model ARMA(p={p},q={q}) si predictii')
plt.show()

fig.savefig('PS_09_4_1.png', format='png')
fig.savefig('PS_09_4_1.pdf', format='pdf')

# Best ARMA (p=4, q=15)
# Best Time Horizon (m): 71
# Best MSE: 2250581.5399686876

#%%

num_folds = 5
fold_size = N // num_folds

best_mse = float('inf')
best_p = None
best_m = None

for p_candidate in range(1, 20):
    for q_candidate in range(1, 20):
        for m_candidate in range(1, fold_size // 2 - max(p_candidate, q_candidate)):
            mse_sum = 0
            
            for fold in range(num_folds):
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size
                
                fold_set = time_series[start_idx:end_idx]
                
                train_set = fold_set[: len(fold_set) // 2]
                validation_set = fold_set[len(fold_set) // 2 : ]
                
                x_AR, x_MA = estimate_arma_coefficients(train_set, p_candidate, q_candidate, m_candidate, miu)
                predictions = predict_arma_model(validation_set, p_candidate, q_candidate, m_candidate, x_AR, x_MA, miu)
                
                mse_fold = np.mean((validation_set[max(p_candidate, q_candidate):] - predictions[max(p_candidate, q_candidate):]) ** 2)
                mse_sum += mse_fold
            
            average_mse = mse_sum / num_folds
            
            if average_mse < best_mse:
                best_mse = average_mse
                best_p = p_candidate
                best_q = q_candidate
                best_m = m_candidate

print(f"Best ARMA (p={best_p}, q={best_q})")
print(f"Best Time Horizon (m): {best_m}")
print(f"Best MSE: {best_mse}")

#%% 

x_AR, x_MA = estimate_arma_coefficients(time_series, best_p, best_q, best_m, miu)
best_predictions = predict_arma_model(time_series, best_p, best_q, best_m, x_AR, x_MA, miu)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x, time_series, label='Serie de timp originala')
ax.plot(x, best_predictions, label=f'Best ARMA(p={best_p},q={best_q}) predictii', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle('Best ARMA(p={best_p},q={best_q}) si predictii')
plt.show()

fig.savefig('PS_09_4_2.png', format='png')
fig.savefig('PS_09_4_2.pdf', format='pdf')

import statsmodels.api as sm

order = (p, 2, q)
model = sm.tsa.ARIMA(time_series, order=order)
results = model.fit()


print(results.summary())

plt.plot(time_series, label='Seria de timp originala')
plt.plot(results.fittedvalues, color='red', label='Seria de timp ajustata cu ARIMA(p={p}, d={2}, q={q})')
plt.legend(loc='upper left')
plt.show()
