"""Read keras datasets"""
from dataclasses import dataclass

from src.domain.models.datasets import Mnist
from src.infra.datasets import MnistReader


@dataclass
class MnistDataSet:
    """Mnist dataset
    """

    def load_dataset(self) -> Mnist:
        """load dataset"""
        reader = MnistReader()

        return reader.load()
