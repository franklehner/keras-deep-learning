#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import List, Optional, Tuple

import click
import numpy as np
from keras import layers
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.datasets import mnist

from src.domain.gan import GAN

Models = Tuple[Model, Model, Model]
Data = Tuple[np.ndarray, np.ndarray]
Params = Tuple[int, int, int, int, str]


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--filepath",
    "-f",
    required=False,
    help="Path to the model",
)
@click.option(
    "--digit",
    "-d",
    required=False,
    help="Digit to generate",
)
def cli(filepath: Optional[str], digit: Optional[int]):
    """Client"""
    if filepath:
        generator = load_model(filepath=filepath)
        class_label = None
        if digit is not None:
            class_label = digit
        test_generator(generator=generator, class_label=class_label)
    else:
        build_and_train_models()


def test_generator(generator: Model, class_label: Optional[int] = None):
    """test the generator"""
    gan = GAN(model_name="test_outputs")
    noise_input = np.random.uniform(
        -1.0,
        1.0,
        size=[16, 100],
    )
    step = 0
    if class_label is None:
        num_labels = 10
        noise_label = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_label = np.zeros((16, 10))
        noise_label[:, class_label] = 1
        step = class_label

    gan.plot_images(
        generator=generator,
        noise_input=noise_input,
        noise_label=noise_label,
        show=True,
        step=step,
    )


def build_and_train_models():
    """build and train model"""
    (x_train, y_train), (_, _) = mnist.load_data()
    gan = GAN(
        model_name="acgan_mnist",
        batch_size=64,
        latent_size=100,
        train_steps=40000,
        image_size=x_train.shape[1],
    )
    x_train = np.reshape(x_train, [-1, gan.image_size, gan.image_size, 1])
    x_train = x_train.astype("float32") / 255
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)
    input_shape = (gan.image_size, gan.image_size, 1)
    label_shape = (num_labels,)
    inputs = layers.Input(shape=input_shape, name="discriminator_input")
    discriminator = gan.discriminator(inputs=inputs, labels=num_labels)
    optimizer = RMSprop(learning_rate=2e-4)
    loss = ["binary_crossentropy", "categorical_crossentropy"]
    discriminator.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    discriminator.summary()
    input_shape = (gan.latent_size,)
    inputs = layers.Input(shape=input_shape, name="z_input")
    labels = layers.Input(shape=label_shape, name="labels")
    generator = gan.generator(
        inputs=inputs,
        labels=labels,
    )
    generator.summary()
    optimizer = RMSprop(learning_rate=2e-4 * 0.5)
    discriminator.trainable = False
    adversarial = Model(
        inputs=[inputs, labels],
        outputs=discriminator(generator([inputs, labels])),
        name=gan.model_name,
    )
    adversarial.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    adversarial.summary()

    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)

    train(models=models, data=data, network=gan, num_labels=num_labels)


def train(models: Models, data: Data, network: GAN, num_labels: int):
    """train model"""
    generator, discriminator, adversarial = models

    for i in range(network.train_steps):
        metrics = train_discriminator(
            generator=generator,
            discriminator=discriminator,
            num_labels=num_labels,
            network=network,
            data=data,
        )
        fmt = "%d: [disc loss: %f, srcloss: %f,"
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f"
        log = fmt % (i, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])
        metrics = train_adversarial(
            adversarial=adversarial,
            num_labels=num_labels,
            network=network,
        )
        fmt = "%s: [advr loss: %f, srcloss: %f,"
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f"
        log = fmt % (log, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4])
        print(log)

        if (i + 1) % network.save_intervals == 0:
            network.plot_images(
                generator=generator,
                noise_input=np.random.uniform(-1.0, 1.0, size=[16, network.latent_size]),
                noise_label=np.eye(num_labels)[np.arange(0, 16) % num_labels],
                show=False,
                step=(i + 1),
            )

    generator.save(network.model_name + ".keras")


def train_discriminator(
    generator: Model,
    discriminator: Model,
    num_labels: int,
    network: GAN,
    data: Data,
) -> List[float]:
    """generate fake data"""
    x_train, y_train = data
    rand_indexes = np.random.randint(0, x_train.shape[0], size=network.batch_size)
    fake_labels = np.eye(num_labels)[np.random.choice(num_labels, network.batch_size)]
    fake_images = generator.predict(
        [
            np.random.uniform(
                -1.0,
                1.0,
                size=[network.batch_size, network.latent_size],
            ),
            fake_labels,
        ],
    )
    x = np.concatenate((x_train[rand_indexes], fake_images))
    labels = np.concatenate((y_train[rand_indexes], fake_labels))
    y = np.ones([2 * network.batch_size, 1])
    y[network.batch_size:, :] = 0
    metrics = discriminator.train_on_batch(x, [y, labels])

    return metrics


def train_adversarial(adversarial: Model, num_labels: int, network: GAN) -> List[float]:
    """prepare adversarial"""
    noise = np.random.uniform(
        -1.0,
        1.0,
        size=[network.batch_size, network.latent_size],
    )
    fake_labels = np.eye(num_labels)[
        np.random.choice(num_labels, network.batch_size)
    ]
    y = np.ones([network.batch_size, 1])
    metrics = adversarial.train_on_batch([noise, fake_labels], [y, fake_labels])

    return metrics


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
