import numpy as np
from ..base_model import LinearModel
from ..losses import MSE, MAE, LogCosh, BaseLoss

class LinearRegression(LinearModel):
    def __init__(self,
                 loss: str='mse',
                 **kwargs) -> None:
        super().__init__(loss, **kwargs)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass of the linear regression model."""
        return X @ self.weights + self.bias

    def _create_loss(self) -> BaseLoss:
        losses_map = {
            'mse': MSE,
            'mae': MAE,
            'logcosh': LogCosh,
        }

        creator = losses_map.get(self.loss.lower())
        if creator is None:
            raise ValueError(f"Unknown loss or not allowed for linear regression loss: {self.loss}")

        return creator()


