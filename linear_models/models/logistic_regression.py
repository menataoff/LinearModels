import numpy as np
from ..core import LinearModel
from ..losses import BaseLoss, BinaryCrossEntropy

class LogisticRegression(LinearModel):
    def __init__(self,
                 loss: str='binarycrossentropy',
                 **kwargs) -> None:
        super().__init__(loss, **kwargs)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return (1 / (1 + np.exp(-z)))

    def _create_loss(self) -> BaseLoss:
        losses_map = {
            'binarycrossentropy': BinaryCrossEntropy,
            'bce': BinaryCrossEntropy
        }

        creator = losses_map.get(self.loss.lower())
        if creator is None:
            raise ValueError(f"Unknown loss or not allowed for logistic regression loss: {self.loss}")

        return creator()