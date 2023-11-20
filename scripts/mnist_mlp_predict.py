#!/usr/bin/env python
"""This script ...
"""
import logging
import os

import click

from src.domain.recognize import Recognizer

_log = logging.getLogger(__name__)


BASE_PATH = "data/"


@click.command()
@click.option(
    "--model-name",
    "-m",
    default="mlp-mnist-sequential.keras",
    help="Filepath to the model",
)
@click.option(
    "--size",
    default=2,
    help="size of the labels to predict",
)
def cli(model_name: str, size: int):
    """Client"""
    filename = os.path.join(BASE_PATH, model_name)
    model = Recognizer(path=filename)
    result = model.recognize_random_sample(size=size)
    if result:
        classifications, real_labels = result
        for classification, real_label in zip(classifications, real_labels):
            print(
                f"Classification: {classification}",
                f"Real Label: {real_label}\t\t{classification == real_label}",
            )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
