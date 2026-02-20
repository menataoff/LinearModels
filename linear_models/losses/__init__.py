"""
losses - module for computing loss
"""

from .base_loss import BaseLoss
from .mse import MSE
from .mae import MAE
from .logcosh import LogCosh
from .binarycrossentropy import BinaryCrossEntropy

__all__ = ['BaseLoss', 'MSE', 'MAE', 'LogCosh', 'BinaryCrossEntropy']

__version__ = '0.1.0'