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

fig.savefig('PS_08_1.png', format='png')
fig.savefig('PS_08_1.pdf', format='pdf')

### b)

lags = np.arange(0, N)

def autocorr(x, lags):
    mean_x = np.mean(x)
    var_x = np.var(x)
    xp = x - mean_x
    acorr = np.correlate(xp, xp, 'full')[len(x) - 1:] / (var_x * len(x))
    
    acorr_manual = np.array([1. if l == 0 else np.sum(xp[l:] * xp[:-l]) / (var_x * len(x)) for l in lags])
    
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(lags, acorr_manual, color="orange", label="Autocorelatie calculata manual")
    ax.plot(lags, acorr, label="Autocorelatie cu np.correlate")
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorelatia')
    ax.grid(True)
    ax.legend()
    
    fig.suptitle('Autocorelatia seriei de timp')
    plt.show()
    
    fig.savefig('PS_08_2.png', format='png')
    fig.savefig('PS_08_2.pdf', format='pdf')

autocorr(time_series, lags)

### c)

p = 2
m = 100

def estimate_ar_coefficients(ts, p, m):
    y = ts[p:p+m]
    
    Y = np.column_stack([ts[i - p : i][::-1] for i in range(p, p + m)])
    Y = Y.T
    
    x = np.linalg.lstsq(Y, y, rcond=None)[0]
    return x

def predict_ar_model(ts, p, m, x):
    n = len(ts)
    predictions = np.zeros(n)
    
    for i in range(p, n):
        x_pred = np.array(ts[i - p : i][::-1])
        
        predictions[i] = np.dot(x, x_pred)
    
    return predictions
    

x_estimated = estimate_ar_coefficients(time_series, p, m)
predictions = predict_ar_model(time_series, p, m, x_estimated)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x, time_series, label='Serie de timp originala')
ax.plot(x, predictions, label=f'Predictii AR(p={p})', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Model AR(p={p}) și predictii')
plt.show()

fig.savefig('PS_08_3.png', format='png')
fig.savefig('PS_08_3.pdf', format='pdf')

lmt = 100

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x[:lmt], time_series[:lmt], label='Serie de timp originala')
ax.plot(x[:lmt], predictions[:lmt], label=f'Predictii AR(p={p})', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle(f'Model AR(p={p}) și predictii')
plt.show()

fig.savefig('PS_08_3_1.png', format='png')
fig.savefig('PS_08_3_1.pdf', format='pdf')

# ### d)

num_folds = 5
fold_size = N // num_folds

best_mse = float('inf')
best_p = None
best_m = None

for p_candidate in range(1, 20):
    for m_candidate in range(1, fold_size // 2 - p_candidate):
        mse_sum = 0
        
        for fold in range(num_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size
            
            fold_set = time_series[start_idx:end_idx]
            
            train_set = fold_set[: len(fold_set) // 2]
            validation_set = fold_set[len(fold_set) // 2 : ]
            
            x_estimated = estimate_ar_coefficients(train_set, p_candidate, m_candidate)
            predictions = predict_ar_model(validation_set, p_candidate, m_candidate, x_estimated)
            
            mse_fold = np.mean((validation_set[p_candidate:] - predictions[p_candidate:]) ** 2)
            mse_sum += mse_fold
        
        average_mse = mse_sum / num_folds
        
        if average_mse < best_mse:
            best_mse = average_mse
            best_p = p_candidate
            best_m = m_candidate

print(f"Best AR Order (p): {best_p}")
print(f"Best Time Horizon (m): {best_m}")
print(f"Best MSE: {best_mse}")

x_estimated = estimate_ar_coefficients(time_series, best_p, best_m)
best_predictions = predict_ar_model(time_series, best_p, best_m, x_estimated)

fig, ax = plt.subplots(1, figsize=(12, 6))
ax.plot(x, time_series, label='Serie de timp originala')
ax.plot(x, best_predictions, label=f'Best AR(p={best_p}, m={best_m}) predictii', linestyle='dashed')
ax.set_xlabel('Timp')
ax.set_ylabel('Valoare')
ax.legend()

fig.suptitle('Best AR model si predictii')
plt.show()

fig.savefig('PS_08_4.png', format='png')
fig.savefig('PS_08_4.pdf', format='pdf')

