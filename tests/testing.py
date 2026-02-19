import numpy as np
import matplotlib.pyplot as plt
from linear_models.models import LinearRegression

np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 3)
true_weights = np.array([2.5, -1.8, 3.0])
true_bias = 1.2
y = X @ true_weights + true_bias + np.random.randn(n_samples) * 0.5

X_train, y_train = X[:700], y[:700]
X_test, y_test = X[700:], y[700:]

configs = [
    {'loss': 'mse', 'opt': 'sgd', 'lr': 0.01, 'batch': None},
    {'loss': 'mse', 'opt': 'adam', 'lr': 0.01, 'batch': 0.3},
    {'loss': 'mae', 'opt': 'adam', 'lr': 0.005, 'batch': 0.5},
    {'loss': 'logcosh', 'opt': 'rmsprop', 'lr': 0.01, 'batch': 0.2, 'gamma': 0.9},
    {'loss': 'mse', 'opt': 'momentum', 'lr': 0.01, 'batch': None, 'beta': 0.9},
    {'loss': 'mse', 'opt': 'nag', 'lr': 0.01, 'batch': 0.4, 'beta': 0.9},
    {'loss': 'mae', 'opt': 'adagrad', 'lr': 0.05, 'batch': None},
]

results = []

for i, cfg in enumerate(configs):
    print(f"\n{'=' * 60}")
    print(f"Тест {i + 1}: loss={cfg['loss']}, opt={cfg['opt']}, batch={cfg['batch']}")
    print('=' * 60)

    model = LinearRegression(
        loss_function=cfg['loss'],
        optimizer=cfg['opt'],
        lr=cfg['lr'],
        **{k: v for k, v in cfg.items() if k not in ['loss', 'opt', 'lr', 'batch']}
    )

    model.fit(X_train, y_train, epochs=300, batch_size=cfg['batch'], verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)

    weight_error = np.mean(np.abs(model.weights - true_weights))
    bias_error = np.abs(model.bias[0] - true_bias)

    final_loss = model.loss_history[-1] if model.loss_history else None

    results.append({
        'config': f"{cfg['loss']}/{cfg['opt']}/batch={cfg['batch']}",
        'train_mse': train_mse,
        'test_mse': test_mse,
        'weight_error': weight_error,
        'bias_error': bias_error,
        'final_loss': final_loss,
        'loss_history': model.loss_history
    })

    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Weight error: {weight_error:.6f}")
    print(f"Bias error: {bias_error:.6f}")
    print(f"True weights: {true_weights}")
    print(f"Learned weights: {model.weights}")
    print(f"True bias: {true_bias:.2f}")
    print(f"Learned bias: {model.bias[0]:.2f}")

print("\n" + "=" * 60)
print("СВОДНЫЙ АНАЛИЗ")
print("=" * 60)

print(f"{'Конфигурация':<40} {'Train MSE':<12} {'Test MSE':<12} {'Weight Err':<10} {'Bias Err':<8}")
print("-" * 82)

best_by_test = min(results, key=lambda x: x['test_mse'])
best_by_weights = min(results, key=lambda x: x['weight_error'])

for r in results:
    config = r['config'][:38] + '..' if len(r['config']) > 38 else r['config']
    marker = ""
    if r == best_by_test:
        marker = " ★ best test"
    elif r == best_by_weights:
        marker = " ☆ best weights"
    print(
        f"{config:<40} {r['train_mse']:<12.6f} {r['test_mse']:<12.6f} {r['weight_error']:<10.6f} {r['bias_error']:<8.6f}{marker}")

print("\n" + "=" * 60)
print("ВИЗУАЛИЗАЦИЯ СХОДИМОСТИ")
print("=" * 60)

plt.figure(figsize=(15, 10))
for i, r in enumerate(results[:4]):
    plt.subplot(2, 2, i + 1)
    plt.plot(r['loss_history'])
    plt.title(r['config'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_comparison.png')
print("Графики сходимости сохранены в 'convergence_comparison.png'")

best_idx = np.argmin([r['test_mse'] for r in results])
print(f"\nЛучшая конфигурация по тестовой MSE: {results[best_idx]['config']}")
print(f"Лучшая тестовая MSE: {results[best_idx]['test_mse']:.6f}")