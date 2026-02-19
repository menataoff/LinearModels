"""
BaseLoss - base class for losses.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss given y_pred and y."""
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradient of loss given y_pred and y."""
        pass
