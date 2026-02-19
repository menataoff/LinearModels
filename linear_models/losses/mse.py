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

        error = y_pred - y

        gradient_weights = (2 / n) * X.T@error
        gradient_bias = (2 / n) * np.sum(error)

        return gradient_weights, gradient_bias


