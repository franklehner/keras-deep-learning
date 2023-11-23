#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Optional

import click

from src.domain.dcgan import DCGAN

_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--filepath",
    "-f",
    required=False,
    help="Load generator h5 model with trained weights",
)
@click.option(
    "--train-steps",
    "-t",
    default=40000,
    help="Number of training steps",
)
def cli(filepath: Optional[str], train_steps: int):
    """Client"""
    dcgan = DCGAN(
        kernel_size=5,
        save_interval=500,
        batch_size=64,
        latent_size=100,
        train_steps=train_steps,
    )
    if filepath:
        generator = dcgan.load_model(filepath=filepath)
        dcgan.test_generator(generator=generator)
    else:
        dcgan.build_and_train_models()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
