"""Autoencoder for cifar-data"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.utils import plot_model
from numpy import ndarray

from src.app.dataset_reader import Cifar10DataSet, Cifar100DataSet
from src.app.net_builder import NetBuilder
from src.app.yaml_reader import NetworkReader
from src.domain.models.datasets import DataSet
from src.domain.models.net_configurations import Layer, Network

Sequence = List[Layer]
InputShape = Tuple[int, ...]
Filters = Tuple[int, ...]
TrainData = Tuple[ndarray, ...]


@dataclass
class CifarAutoencoder:
    """Autoencoder for cifar dataset"""

    model_path: str
    encoder_path: str
    decoder_path: str
    epochs: int = 30
    batch_size: int = 32
    dataset_name: str = "Cifar10"

    def get_dataset(self) -> DataSet:
        """get cifar datasets"""
        if self.dataset_name == "Cifar10":
            return Cifar10DataSet().load_dataset()

        return Cifar100DataSet().load_dataset()

    def get_network(self, params: Dict) -> Network:
        """get network"""
        reader = NetworkReader()
        network = reader.read_network(params=params)

        return network

    def build_network(
        self,
        network: Network,
        freezed: Optional[InputShape] = None,
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

    def create_model(self, params: Dict) -> Model:
        """create model"""
        encoder_network = self.get_network(
            params=params["encoder"],
        )
        enc_inputs, enc_outputs, enc_builder = self.build_network(
            network=encoder_network,
        )
        encoder = Model(
            inputs=enc_inputs,
            outputs=enc_outputs,
            name="encoder",
        )
        encoder.summary()
        plot_model(
            model=encoder,
            to_file=self.encoder_path,
            show_shapes=True,
            show_layer_activations=True,
        )
        decoder_network = self.get_network(
            params=params["decoder"],
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
            to_file=self.decoder_path,
        )
        autoencoder = Model(
            inputs=enc_inputs,
            outputs=decoder(encoder(enc_inputs)),
        )
        autoencoder.summary()
        self.epochs = decoder_network.epochs
        self.batch_size = decoder_network.batch_size

        return autoencoder

    def train(self, model: Model, data: TrainData, theme: str):
        """Train the model"""
        x_train, x_train_gray, x_test, x_test_gray = data
        save_dir = os.path.join(os.getcwd(), "saved_models")
        model_name = theme + ".{epoch:03d}.hd5"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, model_name)
        lr_reducer = ReduceLROnPlateau(
            factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            verbose=1,
            min_lr=0.5e-6,
        )
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
        )
        callbacks = [lr_reducer, checkpoint]

        model.fit(
            x_train_gray,
            x_train,
            validation_data=(x_test_gray, x_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
        )
        model.save(filepath=self.model_path)
