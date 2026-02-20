"""
BinaryCrossEntropy loss
"""
from .base_loss import BaseLoss
import numpy as np

class BinaryCrossEntropy(BaseLoss):
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) #0 and 1 - critical values. Сlipping to avoid log(0
        return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

    def gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]

        error = y_pred - y

        gradient_weights = (1 / n) * X.T@error
        gradient_bias = (1 / n) * np.sum(error)

        return gradient_weights, gradient_bias


