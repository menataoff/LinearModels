import numpy as np
from optimizers import SGD, Adam, Parameter

# Тест SGD
w = np.array([5.0], dtype=np.float32)
param = Parameter(w)
opt = SGD([param], lr=0.1)

for _ in range(100):
    param.grad = np.array([2 * w[0]])  # градиент x²
    opt.step()
    opt.zero_grad()

print(f"SGD: w = {w[0]:.6f}")

# Тест Adam
w2 = np.array([5.0], dtype=np.float32)
param2 = Parameter(w2)
opt2 = Adam([param2], lr=0.1, beta1=0.95, beta2=0.95)

for _ in range(256):
    param2.grad = np.array([2 * w2[0]])
    opt2.step()
    opt2.zero_grad()

print(f"Adam: w = {w2[0]:.6f}")