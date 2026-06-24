import numpy as np
import matplotlib.pyplot as plt

def lwr(x, X, y, tau):
    w = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2))
    theta = np.linalg.inv(X.T @ np.diag(w) @ X) @ (X.T @ np.diag(w) @ y)
    return x @ theta

X = np.linspace(0, 6.5, 100).reshape(-1, 1)
X_bias = np.c_[np.ones(100), X]
y = np.sin(X).flatten() + np.random.normal(0, 0.1, 100)

tau = 0.5
y_pred = [lwr(xi, X_bias, y, tau) for xi in X_bias]

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Training Data', s=20)
plt.plot(X, y_pred, color='blue', label=f'LWR fit (tau={tau})')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
