"""Get datasets from keras
"""
from dataclasses import dataclass

from keras.datasets import mnist, cifar10, cifar100

from src.domain.models.datasets import Mnist


@dataclass
class MnistReader:
    """
    class for the mnist data of keras
    """

    def load(self) -> Mnist:
        """load train and test data from the keras dataset"""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        return DataSet(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


@dataclass
class Cifar10Reader:
    """class of the cifar10 data from keras"""

    def load(self) -> DataSet:
        """load train and test data from cifar10 dataset"""
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        return DataSet(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )


@dataclass
class Cifar100Reader:
    """class of the cifar 100 dataset from keras"""

    def load(self) -> DataSet:
        """load cifar100 data"""
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        return DataSet(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
