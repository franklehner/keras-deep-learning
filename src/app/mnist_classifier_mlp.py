"""Application for mnist classifier mlp
"""
from dataclasses import dataclass
from typing import List, Tuple

from keras.utils import to_categorical
from numpy import ndarray

from src.domain.models.datasets import Mnist
from src.domain.models.net_configurations import Layer, Network
from src.domain.models.neural_network import DenseLayer, NNSequential


@dataclass
class NetBuilderMLP:
    """Builder for fully dense network"""

    model_path: str
    dataset: Mnist
    network: Network

    def prepare_dataset(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Bring dataset in valuable format"""
        image_size = self.dataset.x_train.shape[1]
        y_train = to_categorical(y=self.dataset.y_train)
        y_test = to_categorical(y=self.dataset.y_test)
        x_train = (
            self.dataset.x_train.reshape(-1, image_size * image_size).astype("float32")
            / 255
        )
        x_test = (
            self.dataset.x_test.reshape(-1, image_size * image_size).astype("float32")
            / 255
        )

        return x_train, y_train, x_test, y_test

    def _parse_sequence(self, sequence: List[Layer]) -> NNSequential:
        """parse sequence into a neural net"""
        model = NNSequential(path=self.model_path)
        for layer in sequence:
            if layer.name == "Dense":
                model.add_layer(
                    layer=DenseLayer(
                        units=layer.units,
                        input_dim=layer.input_dim,
                    ),
                )
            elif layer.name == "Activation":
                model.add_activation(activation=layer.activation)
            elif layer.name == "Dropout":
                model.add_dropout(rate=layer.dropout)

        return model

    def parse_network(self) -> NNSequential:
        """parse the network sequence into Neural Net"""
        sequence = self.network.sequence
        model = self._parse_sequence(sequence=sequence)

        return model
