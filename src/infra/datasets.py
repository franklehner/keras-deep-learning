"""Get datasets from keras
"""
from dataclasses import dataclass

from keras.datasets import mnist

from src.domain.models.datasets import Mnist


@dataclass
class MnistReader:
    """
    class for the mnist data of keras
    """

    def load(self) -> Mnist:
        """load train and test data from the keras dataset"""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        return Mnist(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
