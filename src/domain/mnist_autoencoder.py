"""Autoencoder with mnist data"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from keras.models import Model
from keras.utils import plot_model

from src.app.dataset_reader import MnistDataSet
from src.app.net_builder import NetBuilder
from src.app.net_inputs import prepare_dataset
from src.app.yaml_reader import YamlNetwork
from src.domain.models.datasets import DataSet
from src.domain.models.net_configurations import Layer, Network

Sequence = List[Layer]
InputShape = Tuple[int, ...]
Filters = List[int]


@dataclass
class MnistAutoencoder:
    """Autoencoder for mnist dataset"""

    model_path: str

    def get_dataset(self) -> DataSet:
        """get mnist data"""
        mnist_set = MnistDataSet()

        return mnist_set.load_dataset()

    def get_network(self, filepath: str) -> Network:
        """get network from yaml file"""
        reader = YamlNetwork()
        network = reader.read_network_from_yaml(
            network_path=filepath,
        )
        assert isinstance(network, Network)

        return network

    def build_network(
        self, network: Network, freezed: Optional[InputShape] = None,
    ) -> Tuple[Model, Model, NetBuilder]:
        """build network"""
        if freezed is None:
            net_builder = NetBuilder(
                model_path=self.model_path,
                network=network,
            )
        else:
            net_builder = NetBuilder(
                model_path=self.model_path,
                network=network,
                freezed_shape=freezed,
            )
        inputs, outputs = net_builder.parse_sequence_functional(
            sequence=network.sequence,
        )

        return inputs, outputs, net_builder

    def create_model(self) -> Model:
        """create model"""
        encoder_network = self.get_network(
            filepath="data/model_inputs/mnist_encoder.yaml",
        )
        enc_inputs, enc_outputs, enc_builder = self.build_network(network=encoder_network)
        encoder = Model(
            inputs=enc_inputs,
            outputs=enc_outputs,
            name="encoder",
        )
        encoder.summary()
        plot_model(
            model=encoder,
            to_file="data/model_graphs/encoder.png",
            show_shapes=True,
            show_layer_activations=True,
        )
        decoder_network = self.get_network(
            filepath="data/model_inputs/mnist_decoder.yaml",
        )
        dec_inputs, dec_outputs, _ = self.build_network(
            network=decoder_network,
            freezed=enc_builder.freezed_shape,
        )

        decoder = Model(
            inputs=dec_inputs,
            outputs=dec_outputs,
            name="decoder",
        )
        decoder.summary()
        plot_model(
            model=decoder,
            to_file="data/model_graphs/decoder.png",
            show_shapes=True,
            show_layer_activations=True,
        )
        autoencoder = Model(
            inputs=enc_inputs,
            outputs=decoder(encoder(enc_inputs)),
            name="autoencoder",
        )
        autoencoder.summary()
        plot_model(
            model=autoencoder,
            to_file="data/model_graphs/autoencoder.png",
            show_shapes=True,
            show_layer_activations=True,
        )

        return autoencoder

    def run(self):
        """run autoencoder"""
        data = self.get_dataset()
        model = self.create_model()
        model.compile(
            loss="mse",
            optimizer="adam",
        )
        x_train, _, x_test, _ = prepare_dataset(
            dataset=data,
        )
        model.fit(
            x_train,
            x_train,
            validation_data=(x_test, x_test),
            epochs=1,
            batch_size=32,
        )
        model.save(filepath=self.model_path)
