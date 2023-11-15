#!/usr/bin/env python
"""This script ...
"""
import logging
import click

from src.domain.mnist_classifier import MnistNet


_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    cnn = MnistNet(
        path="data/mnist_cnn_functional.yaml",
        model_path="data/foo.keras",
    )
    cnn.run(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
