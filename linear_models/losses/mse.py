"""
MSE loss
"""
from .base_loss import BaseLoss
import numpy as np

class MSE(BaseLoss):
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.float32(np.mean((y_pred - y)**2))

    def gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]

        gradient_weights = (2 / n) * X.T@(y_pred - y)
        gradient_bias = (2 / n) * np.sum(y_pred - y)

        return gradient_weights, gradient_bias


