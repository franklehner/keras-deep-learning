#!/usr/bin/env python
"""Classify mnist data
"""
import logging

import click

from src.app.yaml_reader import YamlNetwork
from src.app.dataset_reader import MnistDataSet
from src.domain.mnist_mlp import MLP

_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--filepath",
    default="data/mnist_mlp_sequential.yaml",
    help="Filepath to network configuration",
)
@click.option(
    "--model-path",
    default="data/mnist_mlp_sequential.keras",
    help="Filepath to save the model",
)
def cli(filepath: str, model_path: str):
    """Client"""
    yaml_network = YamlNetwork()
    mnist_dataset = MnistDataSet()
    mlp = MLP(
        path=filepath,
        model_path=model_path,
        yaml_network=yaml_network,
        mnist_dataset=mnist_dataset,
    )
    mlp.run()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
