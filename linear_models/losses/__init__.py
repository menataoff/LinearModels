"""
losses - module for computing loss
"""

from .base_loss import BaseLoss
from .mse import MSE
from .mae import MAE
from .logcosh import LogCosh

__all__ = ['BaseLoss', 'MSE', 'MAE', 'LogCosh']

__version__ = '0.1.0'