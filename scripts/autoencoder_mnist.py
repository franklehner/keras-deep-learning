#!/usr/bin/env python
"""This script ...
"""
import logging

import click

from src.domain.mnist_autoencoder import MnistAutoencoder

_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-path",
    default="data/model_outputs/mnist_autoencoder.keras",
    help="Where should the model be written",
)
@click.option(
    "--encoder-path",
    default="data/model_inputs/mnist_encoder.yaml",
    help="From where should the encoder input come",
)
@click.option(
    "--decoder-path",
    default="data/model_inputs/mnist_decoder.yaml",
    help="From where should the decoder input come",
)
def cli(model_path: str, encoder_path: str, decoder_path: str):
    """Client
    """
    net = MnistAutoencoder(
        model_path=model_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    )
    net.run()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
