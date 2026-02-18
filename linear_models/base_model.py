"""
LinearModel - base class for linear models
"""
import numpy as np
from optimizers import SGD

class LinearModel:
    def __init__(self, optimizer: str = "sgd") -> None:
        self.str_optimizer = optimizer
        #TODO: выбор оптимизатора

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError(
            "Must implement compute_gradient"
        )

    def _compute_loss(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError(
            "Must implement compute_gradient"
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 128) -> None:
        self.weights = np.zeros(X.shape[1], dtype=np.float32)
        self.bias = 0.0
        self.optimizer = SGD([self.weights, self.bias], lr = 0.01)

        for epoch in range(epochs):
            predictions = X @ self.weights + self.bias
            loss = self._compute_loss(predictions, y)
            grad_weights, grad_bias = self._compute_gradient(X, y, predictions)  # один вызов
            self.optimizer.params[0].grad = grad_weights
            self.optimizer.params[1].grad = grad_bias
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias