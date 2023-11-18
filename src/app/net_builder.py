"""Application for Mnist classification networks"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from keras.models import Model

from src.domain.models.net_configurations import Layer, Network
from src.domain.models.neural_network import (
    ActivationLayer,
    CNNLayer,
    CNNTransposeLayer,
    Concatenation,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    MaxPoolingLayer,
    NNFunctional,
    NNSequential,
    RNNLayer,
    ShapeLayer,
)

LayerMapper = {
    "Dense": DenseLayer,
    "Conv2D": CNNLayer,
    "Conv2DTranspose": CNNTransposeLayer,
    "SimpleRNN": RNNLayer,
    "MaxPooling": MaxPoolingLayer,
    "Dropout": DropoutLayer,
    "Flatten": FlattenLayer,
    "Activation": ActivationLayer,
    "Shape": ShapeLayer,
}


@dataclass
class NetBuilder:
    """Builder for recurrent network"""

    model_path: str
    network: Network
    freezed_shape: Optional[Tuple[int, ...]] = None

    def concatenate_nets(
        self,
        inputs: List[NNFunctional],
        sequence: List[Layer],
    ) -> NNFunctional:
        """concatenate nets"""
        net = Concatenation(inputs=inputs)
        model = net.concatenate()
        for layer in sequence:
            model = LayerMapper[layer.name](**layer.to_dict()).layer(model)

        return model

    def __add_layer(self, model: Model, layer: Layer, net: NNFunctional) -> Model:
        """add layer"""
        if layer.name == "Shape":
            self.freezed_shape = net.get_shape(
                model=model,
                layer=LayerMapper[layer.name](),
            )

            return model

        if layer.name == "Reshape" and self.freezed_shape is not None:
            model = net.reshape(model=model, input_shape=self.freezed_shape)

            return model

        if (
            layer.name == "Dense"
            and layer.units is None
            and self.freezed_shape is not None
        ):
            layer.units = (
                self.freezed_shape[0] * self.freezed_shape[1] * self.freezed_shape[2]
            )

        branch = net.add_layer(
            model=model,
            layer=LayerMapper[layer.name](**layer.to_dict()),
        )

        return branch

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
                branch = self.__add_layer(
                    model=inputs,
                    layer=layer,
                    net=net,
                )
            else:
                branch = self.__add_layer(
                    model=branch,
                    layer=layer,
                    net=net,
                )

        layer = sequence[-1]
        outputs = net.add_layer(
            model=branch,
            layer=LayerMapper[layer.name](**layer.to_dict()),
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
