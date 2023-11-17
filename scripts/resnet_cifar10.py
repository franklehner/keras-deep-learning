#!/usr/bin/env python
"""This script ...
"""
import logging

import click

from src.domain.cifar10_resnet_trainer import Cifar10Resnet

_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client
    """
    cifar10 = Cifar10Resnet(
        version=1,
        depth=20,
    )
    cifar10.run()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
