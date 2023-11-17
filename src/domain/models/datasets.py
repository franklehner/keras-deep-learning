"""Dataset models
"""
from dataclasses import dataclass

from numpy import ndarray


@dataclass
class DataSet:
    """Train and test data"""
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray
