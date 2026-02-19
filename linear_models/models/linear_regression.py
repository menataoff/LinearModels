import numpy as np

class LinearRegression(LinearModel):
    def __init__(self,
                 loss: str='mse',
                 optimizer: str = "sgd",
                 lr: float = 0.01,
                 weight_decay: float = 0.0,
                 **kwargs) -> None:
        self.loss = loss
        super().__init__(optimizer, lr, weight_decay, **kwargs)

    def _compute_loss(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        if (self.loss=='mse'):