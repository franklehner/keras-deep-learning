"""Read keras datasets"""
from dataclasses import dataclass

from src.domain.models.datasets import DataSet
from src.infra.datasets import Cifar10Reader, Cifar100Reader, MnistReader


@dataclass
class MnistDataSet:
    """Mnist dataset
    """

    def load_dataset(self) -> DataSet:
        """load dataset"""
        reader = MnistReader()

        return reader.load()


@dataclass
class Cifar10DataSet:
    """Cifar10 dataset"""

    def load_dataset(self) -> DataSet:
        """load dataset"""
        reader = Cifar10Reader()

        return reader.load()


@dataclass
class Cifar100DataSet:
    """cifar100 dataset"""

    def load(self) -> DataSet:
        """load dataset"""
        reader = Cifar100Reader()

        return reader.load()
