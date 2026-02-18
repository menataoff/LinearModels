"""
LinearModel - base class for linear models
"""
import numpy as np
from optimizers import SGD, Adam, Adagrad, GradientDescent, Momentum, NAG, RMSProp
from optimizers import Parameter, BaseOptimizer
from abc import ABC, abstractmethod
from typing import Iterable, Union

class LinearModel(ABC):
    def __init__(self,
                 optimizer: str = "sgd",
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 **kwargs) -> None:
        self.str_optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_kwargs = kwargs
        self.loss_history = []

    @abstractmethod
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Method to compute the gradient of the loss function"""
        pass
    @abstractmethod
    def _compute_loss(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """Method to compute the loss function"""
        pass
    @abstractmethod
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Method to compute the forward pass of the model"""
        pass

    def _create_optimizer(self, params: Iterable[Union[np.ndarray, Parameter]]) -> BaseOptimizer:
        optimizer_map = {
            "gd": lambda p: GradientDescent(p, lr=self.lr, weight_decay=self.weight_decay),

            "sgd": lambda p: SGD(p, lr=self.lr, weight_decay=self.weight_decay),

            "momentum": lambda p: Momentum(p, lr=self.lr, weight_decay=self.weight_decay,
                                           beta=self.optimizer_kwargs.get('beta')),

            "nag": lambda p: NAG(p, lr=self.lr, weight_decay=self.weight_decay,
                                 beta=self.optimizer_kwargs.get('beta', 0.9)),

            "adagrad": lambda p: Adagrad(p, lr=self.lr, weight_decay=self.weight_decay),

            "rmsprop": lambda p: RMSProp(p, lr=self.lr, weight_decay=self.weight_decay,
                                         gamma=self.optimizer_kwargs.get('gamma'),
                                         eps=self.optimizer_kwargs.get('eps')),

            "adam": lambda p: Adam(p, lr=self.lr, weight_decay=self.weight_decay,
                                   beta1=self.optimizer_kwargs.get('beta1'),
                                   beta2=self.optimizer_kwargs.get('beta2'),
                                   eps=self.optimizer_kwargs.get('epsilon'))
        }

        creator = optimizer_map.get(self.str_optimizer.lower())
        if creator is None:
            raise ValueError(f"Unknown optimizer: {self.str_optimizer}")

        return creator(params)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 256, verbose: bool = True) -> None:
        self.weights = np.array(np.random.randn(X.shape[1]), dtype=np.float32)
        self.bias = np.array(np.random.randn(1), dtype=np.float32)

        # self.weights = np.zeros(X.shape[1], dtype=np.float32)
        # self.bias = np.array([0.0], dtype=np.float32)

        self.optimizer = self._create_optimizer([self.weights, self.bias])

        for epoch in range(epochs):
            predictions = self._forward(X)
            loss = self._compute_loss(predictions, y)
            self.loss_history.append(loss)

            grad_weights, grad_bias = self._compute_gradient(X, y, predictions)  # один вызов
            self.optimizer.params[0].grad = grad_weights
            self.optimizer.params[1].grad = grad_bias

            self.optimizer.step()
            self.optimizer.zero_grad()

            if (epoch + 1) % (epochs // 10) == 0 :
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, loss: {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias