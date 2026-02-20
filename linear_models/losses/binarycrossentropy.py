"""
BinaryCrossEntropy loss
"""
from .base_loss import BaseLoss
import numpy as np

class BinaryCrossEntropy(BaseLoss):
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

    def _sigmoid(self, y_pred: np.ndarray) -> float:
        return np.float32(1 / (1 + np.exp(-y_pred)))

    def gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]

        error = y_pred - y

        gradient_weights = (2 / n) * X.T@error
        gradient_bias = (2 / n) * np.sum(error)

        return gradient_weights, gradient_bias


