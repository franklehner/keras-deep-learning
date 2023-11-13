#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import List, Optional, Tuple, TypedDict

import click

from src.app.dataset_reader import MnistDataSet
from src.app.yaml_reader import YamlNetwork
from src.domain.mnist_classifier import MnistNet

_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--filepath",
    default="data/mnist_cnn_sequential.yaml",
    help="File path to the network",
)
@click.option(
    "--path",
    default="data/cnn-mnist.keras",
    help="Path to the model",
)
def cli(filepath: str, path: str):
    """Client"""
    yaml_network = YamlNetwork()
    mnist_datset = MnistDataSet()
    cnn = MnistNet(
        path=filepath,
        model_path=path,
        yaml_network=yaml_network,
        mnist_dataset=mnist_datset,
    )

    cnn.run(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
