import numpy as np
import matplotlib.pyplot as plt
from linear_models.models import LinearRegression

np.random.seed(42)
X = np.linspace(-3, 3, 500).reshape(-1, 1)
y_true = np.sin(X).flatten() + 0.5 * np.cos(2 * X).flatten()
y = y_true + np.random.randn(500) * 0.15

X_poly = np.hstack([np.sin(X), np.cos(X), np.sin(2*X), np.cos(2*X)])

model = LinearRegression(
    loss_function='mse',
    optimizer='rMsPrOp',
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
)

model.fit(X_poly, y, epochs=10000, verbose=True)

y_pred = model.predict(X_poly)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(model.loss_history)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X, y, alpha=0.5, s=10, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Model')
plt.plot(X, y_true, 'g--', linewidth=2, label='True Function')
plt.legend()
plt.title('Polynomial Features Regression')
plt.xlabel('X')
plt.ylabel('y')

plt.subplot(1, 3, 3)
residuals = y - y_pred
plt.scatter(X, residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals')
plt.xlabel('X')
plt.ylabel('Residual')

plt.tight_layout()
plt.savefig('polynomial_test.png')
print(f"Final loss: {model.loss_history[-1]:.6f}")
print(f"Number of features: {X_poly.shape[1]}")