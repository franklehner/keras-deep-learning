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
    default="data/mnist_cnn_y.yaml",
    help="Path to the yaml file",
)
@click.option(
    "--path",
    default="data/mnist_cnn_y.keras",
    help="Path to the keras file",
)
def cli(filepath: str, path: str):
    """Client
    """
    net = MnistNet(
        path=filepath,
        model_path=path,
    )
    net.run_with_branches(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
