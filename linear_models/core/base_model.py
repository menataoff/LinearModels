"""
LinearModel - base class for linear models
"""
import numpy as np
from optimizers import SGD, Adam, Adagrad, GradientDescent, Momentum, NAG, RMSProp
from optimizers import Parameter, BaseOptimizer
from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional
from ..losses import BaseLoss

class LinearModel(ABC):
    def __init__(self,
                 loss: str,
                 optimizer: str = "sgd",
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 **kwargs) -> None:
        self.loss = loss
        self.str_optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_kwargs = kwargs
        self.loss_history = []

    @abstractmethod
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Method to compute the forward pass of the model"""
        pass

    @abstractmethod
    def _create_loss(self) -> BaseLoss:
        """
        Method to create the loss function
        """
        pass

    def _create_optimizer(self, params: Iterable[Union[np.ndarray, Parameter]]) -> BaseOptimizer:
        optimizer_map = {
            "gd": lambda p: GradientDescent(p, lr=self.lr, weight_decay=self.weight_decay),

            "sgd": lambda p: SGD(p, lr=self.lr, weight_decay=self.weight_decay),

            "momentum": lambda p: Momentum(p, lr=self.lr, weight_decay=self.weight_decay,
                                           beta=self.optimizer_kwargs.get('beta', 0.9)),

            "nag": lambda p: NAG(p, lr=self.lr, weight_decay=self.weight_decay,
                                 beta=self.optimizer_kwargs.get('beta', 0.9)),

            "adagrad": lambda p: Adagrad(p, lr=self.lr, weight_decay=self.weight_decay,
                                         eps=self.optimizer_kwargs.get('eps', 1e-8)),


            "rmsprop": lambda p: RMSProp(p, lr=self.lr, weight_decay=self.weight_decay,
                                         gamma=self.optimizer_kwargs.get('gamma', 0.95),
                                         eps=self.optimizer_kwargs.get('eps', 1e-8)),

            "adam": lambda p: Adam(p, lr=self.lr, weight_decay=self.weight_decay,
                                   beta1=self.optimizer_kwargs.get('beta1', 0.9),
                                   beta2=self.optimizer_kwargs.get('beta2', 0.999),
                                   eps=self.optimizer_kwargs.get('epsilon', 1e-8))
        }

        creator = optimizer_map.get(self.str_optimizer.lower())
        if creator is None:
            raise ValueError(f"Unknown optimizer: {self.str_optimizer}")

        return creator(params)

    def _train(self, X, y, epochs, verbose):
        for epoch in range(epochs):
            y_pred = self._forward(X)

            loss_error = self.loss_function(y_pred, y)
            grad_weights, grad_bias = self.loss_function.gradient(X, y, y_pred)

            self.optimizer.params[0].grad = grad_weights
            self.optimizer.params[1].grad = grad_bias

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_history.append(loss_error)

            if (epoch + 1) % (epochs // 10) == 0 :
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, loss: {loss_error}")

    def _train_epoch(self, X, y, batch_size_int):
        n_samples = X.shape[0]

        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size_int):
            X_batch = X_shuffled[i:i + batch_size_int]
            y_batch = y_shuffled[i:i + batch_size_int]

            y_pred = self._forward(X_batch)
            grad_weights, grad_bias = self.loss_function.gradient(X_batch, y_batch, y_pred)

            self.optimizer.params[0].grad = grad_weights
            self.optimizer.params[1].grad = grad_bias

            self.optimizer.step()
            self.optimizer.zero_grad()

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 250, batch_size: float = 0.5, verbose: bool = True) -> None:
        self.weights = np.array(np.random.randn(X.shape[1]), dtype=np.float32)
        self.bias = np.array(np.random.randn(1), dtype=np.float32)

        self.optimizer = self._create_optimizer([self.weights, self.bias])
        self.loss_function = self._create_loss()

        n_samples = X.shape[0]
        if (batch_size is None) or (batch_size == 1.0):
            stoch = False
        elif (batch_size <= 0.0) or (batch_size > 1.0):
            raise ValueError(f"Batch size must be between 0.0 and 1.0, got {batch_size}")
        else:
            stoch = True
            batch_size_int = int(batch_size * n_samples)

        if stoch:
            for epoch in range(epochs):
                self._train_epoch(X, y, batch_size_int)
                y_pred = self._forward(X)
                loss_error = self.loss_function(y_pred, y)
                self.loss_history.append(loss_error)
                if (epoch + 1) % (epochs // 10) == 0:
                    if verbose:
                        print(f"Epoch {epoch + 1}/{epochs}, loss: {loss_error}")
        else:
            self._train(X, y, epochs, verbose)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)