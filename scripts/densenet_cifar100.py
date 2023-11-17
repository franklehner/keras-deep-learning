#!/usr/bin/env python
"""This script ...
"""
import logging

import click

from src.domain.cifar100_densenet_trainer import Cifar100DenseNet

_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    net = Cifar100DenseNet(
        depth=100,
        growth_rate=12,
        num_dense_blocks=3,
        batch_size=32,
        epochs=200,
    )

    net.run()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
