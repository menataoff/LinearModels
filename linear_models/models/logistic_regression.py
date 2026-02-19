import numpy as np
from ..core import LinearModel

class LogisticRegression(LinearModel):
    def __init__(self,
                 loss: str='binarycrossentropy',
                 **kwargs) -> None:
        super().__init__(loss, **kwargs)