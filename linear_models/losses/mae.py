"""
MAE loss
"""
from .base_loss import BaseLoss
import numpy as np

class MAE(BaseLoss):
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.float32(np.mean(np.abs(y_pred - y)))

    def gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]

        sign_error = np.sign(y_pred - y)

        gradient_weights = (1 / n) * X.T @ sign_error
        gradient_bias = (1 / n) * np.sum(sign_error)

        return gradient_weights, gradient_bias