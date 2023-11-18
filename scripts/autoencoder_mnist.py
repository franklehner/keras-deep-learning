#!/usr/bin/env python
"""This script ...
"""
import logging

import click

from src.domain.mnist_autoencoder import MnistAutoencoder

_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    net = MnistAutoencoder(
        model_path="data/model_outputs/mnist_autoencoder.keras",
    )
    net.run()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
