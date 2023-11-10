#!/usr/bin/env python
"""This script ...
"""
import os
import logging
import click
import numpy as np

import src.app.mnist_classifier_mlp as mlp_cls


_log = logging.getLogger(__name__)


BASE_PATH = "data/"


@click.command()
@click.option(
    "--model-name",
    "-m",
    default="mlp-mnist.keras",
    help="Filepath to the model",
)
def cli(model_name: str):
    """Client
    """
    filename = os.path.join(BASE_PATH, model_name)
    mnist = mlp_cls.load_mnist()
    indexes = np.random.randint(0, mnist.x_test.shape[0], size=10)
    test_images = mnist.x_test[indexes]
    images = test_images.reshape(-1, 28, 28)
    labels = mnist.y_test[indexes]
    model = mlp_cls.load_model(filepath=filename)
    for i in range(len(indexes)):
        mlp_cls.plot_image(image=images[i])
        print(f"Real label: {np.argmax(labels[i])}")
        result = model.predict(image=test_images[i].reshape(-1, test_images[i].shape[0]))
        print(f"Predicted: {np.argmax(result)}")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
