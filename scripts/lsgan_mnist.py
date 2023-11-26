#!/usr/bin/env python
"""This script ...
"""
import logging
import click
from typing import Optional
from keras import layers
from keras.optimizers import RMSprop
from keras.models import Model, load_model
from keras.datasets import mnist
import numpy as np

from src.domain.gan import GAN


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--filepath",
    "-f",
    required=False,
    help="Path to the model",
)
def cli(filepath: Optional[str]):
    """Client
    """
    if filepath:
        gan = GAN()
        generator = load_model(filepath=filepath)
        gan.test_generator(generator=generator)
    else:
        build_and_train_models()


def build_and_train_models():
    """build and train model"""
    (x_train, _), (_, _) = mnist.load_data()
    gan = GAN(
        image_size=x_train.shape[1],
        model_name="lsgan_mnist",
        latent_size=100,
        batch_size=64,
        train_steps=40000,
        kernel_size=5,
    )
    x_train = np.reshape(x_train, [-1, gan.image_size, gan.image_size, 1])
    x_train = x_train.astype("float32") / 255
    input_shape = (gan.image_size, gan.image_size, 1)
    lr = 6e-8
    inputs = layers.Input(shape=input_shape, name="discriminator_input")
    discriminator = gan.discriminator(inputs=inputs, activation=None)
    optimizer = RMSprop(learning_rate=lr)
    discriminator.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    discriminator.summary()

    input_shape = (gan.latent_size,)
    inputs = layers.Input(shape=input_shape, name="z_input")
    generator = gan.generator(inputs=inputs)
    generator.summary()

    optimizer = RMSprop(learning_rate=lr * 0.5)
    discriminator.trainable = False
    adversarial = Model(
        inputs=inputs,
        outputs=discriminator(generator(inputs)),
        name=gan.model_name,
    )
    adversarial.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    adversarial.summary()
    models = (generator, discriminator, adversarial)
    gan.train(models=models, x_train=x_train)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
