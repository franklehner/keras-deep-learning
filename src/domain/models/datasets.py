"""Dataset models
"""
from dataclasses import dataclass
from numpy import ndarray


@dataclass
class Mnist:
    """Mnist dataset
    """
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray
