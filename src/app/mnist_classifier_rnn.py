"""Application for recurrent networks"""
from dataclasses import dataclass
from typing import List, Tuple

from keras.utils import to_categorical
from numpy import ndarray

from src.domain.models.datasets import Mnist
from src.domain.models.net_configurations import Layer, Network
from src.domain.models.neural_network import DenseLayer, NNSequential, RNNLayer


@dataclass
class NetBuilderRNN:
    """Builder for recurrent network"""

    model_path: str
    dataset: Mnist
    network: Network

    def prepare_dataset(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Bring the dataset in a valuable format"""
        image_size = self.dataset.x_train.shape[1]
        y_train = to_categorical(y=self.dataset.y_train)
        y_test = to_categorical(y=self.dataset.y_test)
        x_train = (
            self.dataset.x_train.reshape(-1, image_size, image_size).astype("float32")
            / 255
        )
        x_test = (
            self.dataset.x_test.reshape(-1, image_size, image_size).astype("float32")
            / 255
        )

        return x_train, y_train, x_test, y_test

    def _parse_sequence(self, sequence: List[Layer]) -> NNSequential:
        """parse network sequence"""
        model = NNSequential(path=self.model_path)
        for layer in sequence:
            if layer.name == "SimpleRNN":
                model.add_layer(
                    layer=RNNLayer(
                        units=layer.units,
                        dropout=layer.dropout,
                        input_shape=layer.input_shape,
                    ),
                )
            elif layer.name == "Dense":
                model.add_layer(
                    layer=DenseLayer(
                        units=layer.units,
                        input_dim=layer.input_dim,
                    ),
                )
            elif layer.name == "Activation":
                model.add_activation(activation=layer.activation)

        return model

    def parse_network(self) -> NNSequential:
        """parse the network"""
        sequence = self.network.sequence
        model = self._parse_sequence(sequence=sequence)

        return model
