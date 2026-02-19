"""
LogCosh loss
"""
from .base_loss import BaseLoss
import numpy as np

class LogCosh(BaseLoss):
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.float32(np.mean(np.log(np.cosh(y_pred - y))))

    def gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]

        tanh_error = np.tanh(y_pred - y)

        gradient_weights = (1 / n) * X.T@tanh_error
        gradient_bias = (1 / n) * np.sum(tanh_error)

        return gradient_weights, gradient_bias


