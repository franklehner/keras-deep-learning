#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import List, Optional, Tuple, TypedDict

import click

import src.app.mnist_classifier_cnn as cnn
from src.app.mnist_classifier_cnn import ModelTrainer

_log = logging.getLogger(__name__)


class ModelParams(TypedDict):
    """Model parameters"""

    layers: List[cnn.Layer]
    activations: List[Optional[str]]
    dropouts: List[Optional[float]]
    max_pools: List[Tuple[int, int]]
    batch_size: int
    epochs: int
    model_path: str
    mnist: cnn.Mnist


@click.command()
@click.option(
    "--kernel-size",
    default=3,
    help="Size of the kernel",
)
@click.option(
    "--filters",
    default=64,
    help="Count of the filters",
)
@click.option(
    "--epochs",
    default=10,
    help="Number of iterations",
)
@click.option(
    "--batch-size",
    default=128,
    help="Size of batches",
)
@click.option(
    "--model-path",
    default="data/cnn-mnist.keras",
    help="Path to the model",
)
def cli(kernel_size: int, filters: int, epochs: int, batch_size: int, model_path: str):
    """Client"""
    mnist = cnn.load_mnist()
    conv_layer_count = 3
    dense_layer_count = 1
    num_labels = mnist.num_labels
    layer_count = conv_layer_count + dense_layer_count
    params = cnn.Conv2DParams(
        kernel_size=kernel_size,
        filters=filters,
        activation="relu",
        input_shape=(mnist.image_size, mnist.image_size, 1),
    )
    model_params = ModelParams(
        layers=create_layers(
            conv_layer_count=conv_layer_count,
            dense_layer_count=dense_layer_count,
            params=params,
            units=num_labels,
        ),
        activations=cnn.generate_activations(count=layer_count, last="softmax"),
        dropouts=cnn.generte_dropouts(count=layer_count, last_rate=0.2),
        max_pools=cnn.generate_maxpool(count=2, pool_size=2),
        batch_size=batch_size,
        epochs=epochs,
        model_path=model_path,
        mnist=mnist,
    )
    train_model(params=model_params)


def create_layers(
    conv_layer_count: int,
    dense_layer_count: int,
    params: cnn.Conv2DParams,
    units: int,
) -> List[cnn.Layer]:
    """create layers"""
    layers = cnn.generate_conv2d_layers(count=conv_layer_count, layer_params=params)
    layers.extend(cnn.generate_dense_layers(count=dense_layer_count, units=units))

    return layers


def train_model(params: ModelParams):
    """train_model"""
    trainer = ModelTrainer(
        layers=params["layers"],
        activations=params["activations"],
        dropouts=params["dropouts"],
        max_pools=params["max_pools"],
        model_path=params["model_path"],
        batchsize=params["batch_size"],
    )
    trainer.compile()
    img_size = params["mnist"].image_size
    x_train = params["mnist"].x_train.reshape(-1, img_size, img_size, 1)
    y_train = params["mnist"].y_train
    x_test = params["mnist"].x_test.reshape(-1, img_size, img_size, 1)
    y_test = params["mnist"].y_test
    trainer.fit(x_train=x_train, y_train=y_train, epochs=params["epochs"])
    accuracy = trainer.evaluate(x_test=x_test, y_test=y_test)
    print(f"\n\nTest accuracy: {accuracy}")
    trainer.save()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
