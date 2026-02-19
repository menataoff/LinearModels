from optimizers import SGD, Adam
from linear_models.models import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# Создаем данные
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Обучаем модель
model = LinearRegression(loss='mse', lr=0.01)
model.fit(X, y.flatten(), epochs=500)

# Визуализируем
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model.loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, model.predict(X), 'r-', label='Model')
plt.legend()
plt.title('Linear Regression Fit')

plt.tight_layout()
plt.savefig('linear_regression_test.png')  # сохраняем
print("График сохранен в linear_regression_test.png")

