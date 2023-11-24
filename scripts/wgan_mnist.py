#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import Tuple

import click
import numpy as np
from keras import backend as k
from keras import layers
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.optimizers.legacy import RMSprop

from src.domain.gan import GAN

_log = logging.getLogger(__name__)


Models = Tuple[Model, Model, Model]
Parameters = Tuple[GAN, float, float]


@click.command()
@click.option(
    "--filename",
    "-f",
    required=False,
    help="Path to the model",
)
def cli(filename):
    """Client"""
    if filename is not None:
        generator = load_model(filepath=filename)
        gan = GAN(model_name="test_outputs")
        gan.test_generator(generator=generator)
    else:
        build_and_train_models()


def train(models: Models, x_train: np.ndarray, gan: GAN):
    """train model"""
    generator, discriminator, adversarial = models
    for i in range(gan.train_steps):
        score = [0.0, 0.0]
        for _ in range(int(5)):
            rand_indexes = np.random.randint(0, x_train.shape[0], size=gan.batch_size)
            real_images = x_train[rand_indexes]
            fake_images = generator.predict(
                np.random.uniform(-1.0, 1.0, size=[gan.batch_size, gan.latent_size]),
            )
            real_score = discriminator.train_on_batch(
                real_images,
                np.ones([gan.batch_size, 1]),
            )
            fake_score = discriminator.train_on_batch(
                fake_images,
                np.ones([gan.batch_size, 1]) * (-1),
            )
            score[0] += 0.5 * (real_score[0] + fake_score[0])
            score[1] += 0.5 * (real_score[1] + fake_score[1])
            for layer in discriminator.layers:
                layer.set_weights(
                    weights=[
                        np.clip(weight, -0.01, 0.01) for weight in layer.get_weights()
                    ],
                )

        score[0] /= 5
        score[1] /= 5
        log = f"{i}: [discriminator loss: {score[0]}, acc: {score[1]}]"
        score = adversarial.train_on_batch(
            np.random.uniform(-1.0, 1.0, size=[gan.batch_size, gan.latent_size]),
            np.ones([gan.batch_size, 1]),
        )
        log = f"{log}: [adversarial loss: {score[0]}, acc: {score[1]}]"
        print(log)
        if (i + 1) % gan.save_intervals == 0:
            gan.plot_images(
                generator=generator,
                noise_input=np.random.uniform(-1.0, 1.0, size=[16, gan.latent_size]),
                show=False,
                step=(i + 1),
            )

    generator.save(gan.model_name + ".keras")


def wasserstein_loss(y_label, y_pred):
    """wasserstein loss"""
    return -k.mean(y_label * y_pred)


def build_and_train_models():
    """build and train"""
    (x_train, _), (_, _) = mnist.load_data()
    gan = GAN(
        model_name="wgan_mnist",
        image_size=x_train.shape[1],
    )
    x_train = np.reshape(x_train, [-1, gan.image_size, gan.image_size, 1])
    x_train = x_train.astype("float32") / 255
    lr = 5e-5
    input_shape = (gan.image_size, gan.image_size, 1)
    inputs = layers.Input(shape=input_shape, name="discriminator_input")
    discriminator = gan.discriminator(inputs=inputs, activation="linear")
    optimizer = RMSprop(learning_rate=lr)
    discriminator.compile(
        loss=wasserstein_loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    discriminator.summary()
    input_shape = (gan.latent_size,)
    inputs = layers.Input(shape=input_shape, name="z_input")
    generator = gan.generator(inputs=inputs)
    generator.summary()
    discriminator.trainable = False
    adversarial = Model(
        inputs=inputs,
        outputs=discriminator(generator(inputs)),
        name=gan.model_name,
    )
    adversarial.compile(
        loss=wasserstein_loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    adversarial.summary()
    models = (generator, discriminator, adversarial)
    train(models=models, x_train=x_train, gan=gan)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
