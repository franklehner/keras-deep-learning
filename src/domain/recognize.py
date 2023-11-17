"""Load model and recognize"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from src.app.dataset_reader import MnistDataSet
from src.app.model_loader import load_model
from src.domain.models.datasets import DataSet
from src.domain.models.neural_network import NNFunctional, NNSequential

ClassificationColl = Tuple[List[ndarray], List[int]]


@dataclass
class Model:
    """Model"""

    path: str

    def load_model(self) -> Optional[Union[NNFunctional, NNSequential]]:
        """load model"""
        model = load_model(path=self.path)

        return model

    def load_dataset(self) -> DataSet:
        """load dataset"""
        dataset = MnistDataSet().load_dataset()

        return dataset

    def recognize(self, image: ndarray):
        """recognize"""

    def recognize_image(self, images: ndarray) -> Optional[ndarray]:
        """recognize image"""
        model = self.load_model()
        if model is not None:
            return model.predict(images=images)

        return None

    def recognize_random_sample(self, size: int) -> Optional[ClassificationColl]:
        """recognize random samples"""
        model = self.load_model()
        if model is None:
            return None

        dataset = self.load_dataset()
        indexes = np.random.randint(0, dataset.x_test.shape[0], size=size)
        images = dataset.x_test[indexes].astype("float32") / 255
        images = images.reshape(-1, images.shape[1] * images.shape[1])
        predicted = model.predict(images=images)
        classifications = [pred.argmax() for pred in predicted]
        real_labels = list(dataset.y_test[indexes])

        return classifications, real_labels
