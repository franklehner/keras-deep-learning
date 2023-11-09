#!/usr/bin/env python
"""Classify mnist data
"""
import logging

import click

import src.app.mnist_classifier_mlp as mlp

_log = logging.getLogger(__name__)


@click.command()
def cli():
    """Client"""
    count = 3
    mnist = mlp.load_mnist()
    layers = mlp.generate_layers(
        count=count,
        layer="Dense",
        units=256,
        num_labels=mnist.num_labels,
    )
    activations = mlp.generate_activations(
        count=3,
        activation="relu",
        last="softmax",
    )
    dropout_count = count - 1
    dropouts = mlp.generate_dropouts(count=dropout_count, rate=0.45)
    mlp.run(
        mnist=mnist,
        layers=layers,
        activations=activations,
        dropouts=dropouts,
        model_path="data/mlp-mnist.keras",
        batchsize=128,
        epochs=20,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
