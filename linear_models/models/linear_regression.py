import numpy as np
from ..base_model import LinearModel
from ..losses import *

class LinearRegression(LinearModel):
    def __init__(self,
                 loss_function: str='mse',
                 optimizer: str = "sgd",
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 **kwargs) -> None:
        self.loss_function = loss_function
        super().__init__(optimizer, lr, weight_decay, **kwargs)

    def _compute_loss(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        pass
