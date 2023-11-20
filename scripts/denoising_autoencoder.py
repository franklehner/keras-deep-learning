#!/usr/bin/env python
"""This script ...
"""
import logging

import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.app.net_inputs import prepare_dataset
from src.domain.mnist_autoencoder import MnistAutoencoder
from src.domain.recognize import Recognizer

_log = logging.getLogger(__name__)


@click.group()
def cli():
    """Client"""


@cli.command()
@click.option(
    "--model-path",
    default="data/model_outputs/mnist_denoising_autoencoder_functional.keras",
    help="Where should the model be saved?",
)
@click.option(
    "--encoder-path",
    default="data/model_inputs/mnist_encoder.yaml",
    help="Input path for encoder",
)
@click.option(
    "--decoder-path",
    default="data/model_inputs/mnist_decoder.yaml",
    help="Input path for decoder",
)
def train(model_path: str, encoder_path: str, decoder_path: str):
    """train the model"""
    net = MnistAutoencoder(
        model_path=model_path,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
    )
    data = net.get_dataset()
    autoencoder = net.create_model()
    x_train, _, x_test, _ = prepare_dataset(
        dataset=data,
    )
    noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
    x_train_noisy = x_train + noise
    noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
    x_test_noisy = x_test + noise
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    autoencoder.compile(
        loss="mse",
        optimizer="adam",
    )
    autoencoder.fit(
        x_train_noisy,
        x_train,
        validation_data=(x_test_noisy, x_test),
        epochs=net.epochs,
        batch_size=net.batch_size,
    )

    autoencoder.save(filepath=model_path)


@cli.command()
@click.option(
    "--path",
    default="data/model_outputs/mnist_denoising_autoencoder_functional.keras",
    help="Which model should be loaded?",
)
@click.option(
    "--sigma",
    default=0.5,
    help="How should be the level of the noise",
)
def predict(path: str, sigma: float):
    """predict and plot model"""
    rec = Recognizer(path=path)
    data = rec.load_dataset()
    image_size = data.x_test.shape[1]
    autoencoder = rec.load_model()
    _, _, x_test, _ = prepare_dataset(
        dataset=data,
    )
    x_test = x_test.reshape(-1, image_size, image_size, 1)
    noise = np.random.normal(loc=0.5, scale=sigma, size=x_test.shape)
    x_test_noisy = x_test + noise
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
    if autoencoder is not None:
        x_decoded = rec.recognize_image(images=x_test_noisy)
        if x_decoded is not None:
            rows, cols = 3, 9
            num = rows * cols
            imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
            imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
            imgs = np.vstack([np.hstack(i) for i in imgs])
            imgs = (imgs * 255).astype(np.uint8)
            plt.figure()
            plt.axis("off")
            plt.title(
                "Original images: top rows,\n"
                "Corrupted input: middle rows,\n"
                "Denoised input: third rows",
            )
            plt.imshow(imgs, interpolation="none", cmap="gray")
            Image.fromarray(imgs).save("data/model_graphs/corrupted_and_denoised.png")
            plt.show()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
