#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Optional

import click

from src.domain.cgan import CGAN

_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--filepath",
    "-f",
    required=False,
    help="Load generator h5 model with trained weights",
)
@click.option(
    "--digit",
    "-d",
    required=False,
    help="Specify a digit to generate",
)
def cli(filepath: Optional[str], digit: Optional[int]):
    """Client
    """
    cgan = CGAN()
    if filepath:
        generator = cgan.load_model(filepath=filepath)
        class_label = None
        if digit is not None:
            class_label = digit
        cgan.test_generator(
            generator=generator,
            class_label=class_label,
        )
    else:
        cgan.build_and_train_models()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
