#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from src.domain.cifar10_autoencoder import CifarAutoencoder
from src.domain.models.datasets import DataSet
from src.domain.recognize import Recognizer

_log = logging.getLogger(__name__)


Filters = Tuple[int, ...]


def rgb2gray(rgb: ndarray):
    """Convert colorized images into gray scale"""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


@click.group()
@click.option(
    "--model-path",
    default="data/model_outputs/colorization_autoencoder_cifar10_functional.keras",
    help="Path to the model",
)
@click.option(
    "--encoder-path",
    default="data/model_graphs/cifar_encoder.png",
    help="Path to the enocder",
)
@click.option(
    "--decoder-path",
    default="data/model_graphs/cifar_decoder.png",
    help="Path to the decoder",
)
@click.pass_context
def cli(ctx, model_path: str, encoder_path: str, decoder_path: str):
    """Client"""
    _log.info("%s, %s, %s, %s", model_path, encoder_path, decoder_path, ctx)


@cli.command()
@click.option(
    "--batch-size",
    default=32,
    help="Size of batches",
)
@click.option(
    "--epochs",
    default=30,
    help="Number of iterations of the network",
)
@click.option(
    "--filters",
    default=(64, 128, 256),
    multiple=True,
    help="Filters for the sequence",
)
@click.pass_context
def train(
    ctx,
    batch_size: int,
    epochs: int,
    filters: Filters,
):
    """train autoencoder"""
    cifar_autoencoder = CifarAutoencoder(**ctx.parent.params)
    dataset = cifar_autoencoder.get_dataset()
    kernel_size = 3
    latent_dim = 256
    net_input = create_net_input(
        input_shape=(dataset.x_train.shape[1], dataset.x_train.shape[2], 1),
        conv_params={
            "kernel_size": kernel_size,
            "filters": filters,
            "channels": dataset.x_train.shape[3],
        },
        batch_size=batch_size,
        epochs=epochs,
        latent_dim=latent_dim,
    )
    model = cifar_autoencoder.create_model(params=net_input)
    model.compile(
        loss="mse",
        optimizer="adam",
    )
    x_train, x_train_gray, x_test, x_test_gray = prepare_data(dataset=dataset)
    cifar_autoencoder.train(
        model=model,
        data=(x_train, x_train_gray, x_test, x_test_gray),
        theme="colorized_ae_model",
    )


@cli.command()
@click.pass_context
def predict(ctx):
    """predict and plot images"""
    imgs_dir = "saved_images"
    recognizer = Recognizer(
        path=ctx.parent.params["model_path"],
    )
    dataset = recognizer.load_dataset_cifar10()
    _, _, x_test, x_test_grey = prepare_data(dataset=dataset)
    img_rows, img_cols, channels = x_test_grey.shape[1:]
    imgs = x_test[:100]
    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis("off")
    plt.title("Test color images (Ground Truth)")
    plt.imshow(imgs, interpolation="none")
    plt.savefig(f"{imgs_dir}/test_color.png")
    plt.show()

    x_decoded = recognizer.recognize_image(images=x_test_grey)

    imgs = x_decoded[:100]
    imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis("off")
    plt.title("Colorized test images (predicted)")
    plt.imshow(imgs, interpolation="none")
    plt.savefig(f"{imgs_dir}/colorized.png")
    plt.show()


def create_net_input(
    input_shape: Tuple[int, ...],
    conv_params: Dict,
    batch_size: int,
    epochs: int,
    latent_dim: int,
):
    """create net input"""
    img_rows, img_cols, chn = input_shape
    encoder_sequence: List[Dict] = [
        {
            "Input": {"input_shape": (img_rows, img_cols, chn)},
        }
    ]
    for layer_filter in conv_params["filters"]:
        encoder_sequence.append(
            {
                "Conv2D": {
                    "kernel_size": conv_params["kernel_size"],
                    "filters": layer_filter,
                    "activation": "relu",
                    "strides": 2,
                    "padding": "same",
                },
            },
        )
    encoder_sequence.append(
        {"Shape": {"int_shape": 1}},
    )
    encoder_sequence.append(
        {"Flatten": {"flatten": "flatten"}},
    )
    encoder_sequence.append(
        {"Dense": {"units": latent_dim}},
    )
    decoder_sequence: List[Dict] = [
        {"Input": {"input_shape": (latent_dim,)}},
    ]
    decoder_sequence.append(
        {"Dense": {"units": None}},
    )
    decoder_sequence.append(
        {"Reshape": {"reshape": 1}},
    )
    for layer_filter in conv_params["filters"][::-1]:
        decoder_sequence.append(
            {
                "Conv2DTranspose": {
                    "filters": layer_filter,
                    "kernel_size": conv_params["kernel_size"],
                    "activation": "relu",
                    "strides": 2,
                    "padding": "same",
                },
            },
        )
    decoder_sequence.append(
        {
            "Conv2DTranspose": {
                "filters": conv_params["channels"],
                "kernel_size": conv_params["kernel_size"],
                "activation": "sigmoid",
                "padding": "same",
                "layer_name": "decoder_output",
            },
        },
    )
    net_input = {
        "encoder": {
            "Sequence": encoder_sequence,
            "batch_size": batch_size,
            "epochs": epochs,
            "net": "Autoencoder",
            "network_type": "functional",
        },
        "decoder": {
            "Sequence": decoder_sequence,
            "batch_size": batch_size,
            "epochs": epochs,
            "net": "Autoencoder",
            "network_type": "functional",
        },
    }

    return net_input


def prepare_data(dataset: DataSet) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """prepare data"""
    img_rows = dataset.x_train.shape[1]
    img_cols = dataset.x_train.shape[2]
    channels = dataset.x_train.shape[3]
    x_train_gray = rgb2gray(dataset.x_train)
    x_test_gray = rgb2gray(dataset.x_test)
    x_train = dataset.x_train.astype("float32") / 255
    x_test = dataset.x_test.astype("float32") / 255
    x_train_gray = x_train_gray.astype("float32") / 255
    x_test_gray = x_test_gray.astype("float32") / 255
    x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, channels))
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    x_train_gray = x_train_gray.reshape((x_train_gray.shape[0], img_rows, img_cols, 1))
    x_test_gray = x_test_gray.reshape((x_test_gray.shape[0], img_rows, img_cols, 1))

    return x_train, x_train_gray, x_test, x_test_gray


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
