"""
LinearModels - pure NumPy implementation of module linear_models.
"""
from .core import LinearModel
from .models import LinearRegression
from .losses import MSE, MAE, LogCosh

# Явно указываем, что экспортируется
__all__ = [
    'LinearModel',
    'LinearRegression',
    'MSE',
    'MAE',
    'LogCosh',
]

__version__ = "0.1.0"
__author__ = "menataoff"