"""Application for Mnist classification networks"""
from dataclasses import dataclass
from typing import List, Tuple, Union

from keras.utils import to_categorical
from numpy import ndarray

from src.domain.models.datasets import DataSet
from src.domain.models.net_configurations import Layer, Network
from src.domain.models.neural_network import (
    ActivationLayer,
    CNNLayer,
    Concatenation,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    MaxPoolingLayer,
    NNFunctional,
    NNSequential,
    RNNLayer,
)

LayerMapper = {
    "Dense": DenseLayer,
    "Conv2D": CNNLayer,
    "SimpleRNN": RNNLayer,
    "MaxPooling": MaxPoolingLayer,
    "Dropout": DropoutLayer,
    "Flatten": FlattenLayer,
    "Activation": ActivationLayer,
}


@dataclass
class NetBuilder:
    """Builder for recurrent network"""

    model_path: str
    dataset: DataSet
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

    def concatenate_nets(
        self, inputs: List[NNFunctional], sequence: List[Layer],
    ) -> NNFunctional:
        """concatenate nets"""
        net = Concatenation(inputs=inputs)
        model = net.concatenate()
        for layer in sequence:
            model = LayerMapper[layer.name](**layer.to_dict()).layer(model)

        return model

    def parse_sequence_functional(
        self,
        sequence: List[Layer],
    ) -> Tuple[NNFunctional, NNFunctional]:
        """parse into functional API"""
        net = NNFunctional(path=self.model_path)
        if sequence[0].name == "Input":
            inputs = net.model_input(input_shape=sequence[0].input_shape)

        for idx, layer in enumerate(sequence[1:-1]):
            if idx == 0:
                branch = net.add_layer(
                    model=inputs, layer=LayerMapper[layer.name](**layer.to_dict()),
                )
            else:
                branch = net.add_layer(
                    model=branch, layer=LayerMapper[layer.name](**layer.to_dict()),
                )

        layer = sequence[-1]
        outputs = net.add_layer(
            model=branch, layer=LayerMapper[layer.name](**layer.to_dict()),
        )

        return inputs, outputs

    def parse_sequence_sequential(self, sequence: List[Layer]) -> NNSequential:
        """parse network sequence"""
        model = NNSequential(path=self.model_path)
        for layer in sequence:
            model.add_layer(layer=LayerMapper[layer.name](**layer.to_dict()))

        return model

    def parse_network(self) -> Union[NNFunctional, NNSequential]:
        """parse the network"""
        sequence = self.network.sequence
        if self.network.network_type == "sequential":
            return self.parse_sequence_sequential(sequence=sequence)

        net = NNFunctional(path=self.model_path)
        inputs, outputs = self.parse_sequence_functional(sequence=sequence)
        model = net.generate_model(inputs=inputs, outputs=outputs)

        return model
