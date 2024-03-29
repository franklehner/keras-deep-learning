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
    default="data/mnist_rnn_sequential.yaml",
    help="Filepath to network",
)
@click.option(
    "--path",
    default="data/mnist_rnn_sequential.keras",
    help="Filepath to save the model",
)
def cli(filepath: str, path: str):
    """Client"""
    rnn = MnistNet(
        path=filepath,
        model_path=path,
    )
    rnn.run(
        loss="categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
    )

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
