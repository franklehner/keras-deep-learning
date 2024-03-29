#!/usr/bin/env python
"""This script ...
"""
import logging

import click

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
    cnn = MnistNet(
        path=filepath,
        model_path=path,
    )

    cnn.run(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
