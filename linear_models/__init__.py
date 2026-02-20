"""
LinearModels - pure NumPy implementation of module linear_models.
"""
from .core import LinearModel
from .models import LinearRegression
from .models import LogisticRegression
from .losses import MSE, MAE, LogCosh

# Явно указываем, что экспортируется
__all__ = [
    'LinearModel',
    'LinearRegression',
    'LogisticRegression',
    'MSE',
    'MAE',
    'LogCosh',
]

__version__ = "0.1.0"
__author__ = "menataoff"