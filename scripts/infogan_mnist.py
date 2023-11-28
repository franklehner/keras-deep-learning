#!/usr/bin/env python
"""This script ...
"""
import logging
from typing import List, Optional, Tuple

import click
import numpy as np
from keras import backend as k
from keras import layers
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.utils import to_categorical

from src.domain.gan import GAN

Models = Tuple[Model, Model, Model]
Data = Tuple[np.ndarray, np.ndarray]
Params = Tuple[Optional[int], Optional[float], Optional[float], bool, bool]


_log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--generator-path",
    "-g",
    required=False,
    help="Load generator keras model with trained weights",
)
@click.option(
    "--digit",
    "-d",
    required=False,
    help="Specify a specific digit to generate",
)
@click.option(
    "--code1",
    "-a",
    required=False,
    help="Specify latent code 1",
)
@click.option(
    "--code2",
    "-b",
    required=False,
    help="Specify latent code 2",
)
@click.option(
    "--p1",
    is_flag=True,
    default=False,
    help="Plot digits with code 1 ranging fr -n1 to +n2",
)
@click.option(
    "--p2",
    is_flag=True,
    default=False,
    help="Plot digits with code 2 ranging fr -n1 to +n2",
)
def cli(
    generator_path: Optional[str],
    digit: Optional[int],
    code1: Optional[float],
    code2: Optional[float],
    p1: bool,
    p2: bool,
):  # pylint: disable=too-many-arguments
    """Client"""
    if generator_path:
        generator = load_model(filepath=generator_path)
        params = (digit, code1, code2, p1, p2)
        test_generator(generator=generator, params=params)
    else:
        build_and_train_models()


def mi_loss(c, q_of_c_given_x):
    """Mutual information, Equation:
    assuming H(c) is constant.
    mi_loss = -c * log(Q(c|x))
    """
    return -k.mean(k.sum(c + k.log(q_of_c_given_x + k.epsilon()), axis=1))


def build_and_train_models():
    "Build and train model"
    (x_train, y_train), (_, _) = mnist.load_data()
    gan = GAN(
        model_name="infogan_mnist",
        batch_size=64,
        train_steps=40000,
        save_intervals=500,
        image_size=x_train.shape[1],
        kernel_size=5,
        latent_size=100,
    )
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype("float32") / 255
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)
    lr = 2e-4
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels,)
    code_shape = (1,)
    inputs = layers.Input(shape=input_shape, name="discriminator_input")
    discriminator = gan.discriminator(
        inputs=inputs,
        num_labels=num_labels,
        num_codes=2,
    )
    optimizer = RMSprop(learning_rate=lr)
    loss = [
        "binary_crossentropy",
        "categorical_crossentropy",
        mi_loss,
        mi_loss,
    ]
    loss_weights = [1.0, 1.0, 0.5, 0.5]
    discriminator.compile(
        loss=loss,
        loss_weights=loss_weights,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    discriminator.summary()
    input_shape = (gan.latent_size,)
    inputs = layers.Input(shape=input_shape, name="z_input")
    labels = layers.Input(shape=label_shape, name="labels")
    code1 = layers.Input(shape=code_shape, name="code1")
    code2 = layers.Input(shape=code_shape, name="code2")
    generator = gan.generator(
        inputs=inputs,
        labels=labels,
        codes=[code1, code2],
    )
    generator.summary()
    optimizer = RMSprop(learning_rate=lr * 0.5)
    discriminator.trainable = False
    inputs = [inputs, labels, code1, code2]
    adversarial = Model(
        inputs=inputs,
        outputs=discriminator(generator(inputs)),
        name=gan.model_name,
    )
    adversarial.compile(
        loss=loss,
        loss_weights=loss_weights,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    adversarial.summary()

    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    train(models=models, data=data, num_labels=num_labels, net=gan)


def train(models: Models, data: Data, num_labels, net: GAN):
    """train models"""
    generator, discriminator, adversarial = models
    code_std = 0.5

    for i in range(net.train_steps):
        metrics = train_discriminator(
            generator=generator,
            discriminator=discriminator,
            num_labels=num_labels,
            net=net,
            data=data,
        )
        fmt = "%d: [dis: %.2f, bce: %.2f, ce: %.2f, mi: %.2f, mi: %.2f, acc: %.2f]"
        log = fmt % (
            i,
            metrics[0],
            metrics[1],
            metrics[2],
            metrics[3],
            metrics[4],
            metrics[6],
        )
        metrics = train_adversarial(
            adversarial=adversarial,
            num_labels=num_labels,
            net=net,
        )
        fmt = "%s [adv: %f, bce: %f, ce: %f, mi: %f, mi: %f, acc: %f]"
        log = fmt % (
            log,
            metrics[0],
            metrics[1],
            metrics[2],
            metrics[3],
            metrics[4],
            metrics[6],
        )
        print(log)
        if (i + 1) % net.save_intervals == 0:
            net.plot_images(
                generator=generator,
                noise_input=np.random.uniform(
                    -1.0,
                    1.0,
                    size=[16, net.latent_size],
                ),
                noise_label=np.eye(num_labels)[np.arange(0, 16) % num_labels],
                noise_codes=[
                    np.random.normal(scale=code_std, size=[16, 1]),
                    np.random.normal(scale=code_std, size=[16, 1]),
                ],
                show=False,
                step=(i + 1),
            )

        if (i + 1) % (2 * net.save_intervals) == 0:
            generator.save("data/model_outputs/" + net.model_name + ".keras")

    generator.save(f"data/model_outputs/{net.model_name}.keras")


def train_discriminator(
    generator: Model,
    discriminator: Model,
    num_labels: int,
    net: GAN,
    data: Data,
) -> List[float]:
    """train discriminator"""
    code_std = 0.5
    x_train, y_train = data
    rand_indexes = np.random.randint(
        0,
        x_train.shape[0],
        size=net.batch_size,
    )
    fake_labels = np.eye(num_labels)[np.random.choice(num_labels, net.batch_size)]
    fake_code1 = np.random.normal(
        scale=code_std,
        size=[net.batch_size, 1],
    )
    fake_code2 = np.random.normal(
        scale=code_std,
        size=[net.batch_size, 1],
    )
    inputs = [
        np.random.uniform(-1.0, 1.0, size=[net.batch_size, net.latent_size]),
        fake_labels,
        fake_code1,
        fake_code2,
    ]
    y = np.ones([2 * net.batch_size, 1])
    y[net.batch_size:, :] = 0
    metrics = discriminator.train_on_batch(
        np.concatenate((x_train[rand_indexes], generator.predict(inputs))),
        [
            y,
            np.concatenate((y_train[rand_indexes], fake_labels)),
            np.concatenate(
                (
                    np.random.normal(scale=code_std, size=[net.batch_size, 1]),
                    fake_code1,
                ),
            ),
            np.concatenate(
                (
                    np.random.normal(scale=code_std, size=[net.batch_size, 1]),
                    fake_code2,
                ),
            ),
        ],
    )

    return metrics


def train_adversarial(
    adversarial: Model,
    num_labels: int,
    net: GAN,
) -> List[float]:
    """Train adversarial"""
    code_std = 0.5
    noise = np.random.uniform(
        -1.0,
        1.0,
        size=[net.batch_size, net.latent_size],
    )
    fake_labels = np.eye(num_labels)[np.random.choice(num_labels, net.batch_size)]
    fake_code1 = np.random.normal(
        scale=code_std,
        size=[net.batch_size, 1],
    )
    fake_code2 = np.random.normal(
        scale=code_std,
        size=[net.batch_size, 1],
    )
    y = np.ones([net.batch_size, 1])
    inputs = [noise, fake_labels, fake_code1, fake_code2]
    outputs = [y, fake_labels, fake_code1, fake_code2]
    metrics = adversarial.train_on_batch(inputs, outputs)

    return metrics


def test_generator(
    generator: Model,
    params: Params,
    latent_size: int = 62,
):
    """test generator"""
    label, code1, code2, p1, p2 = params
    step = 0
    if label is None:
        num_labels = 10
        noise_label = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_label = np.zeros((16, 16))
        noise_label[:, label] = 1
        step = label

    if code1 is None:
        noise_code1 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        if p1:
            a = np.linspace(-2, 2, 16)
            a = np.reshape(a, [16, 1])
            noise_code1 = np.ones((16, 1)) * a
        else:
            noise_code1 = np.ones((16, 1)) * code1
        print(noise_code1)

    if code2 is None:
        noise_code2 = np.random.normal(scale=0.5, size=[16, 1])
    else:
        if p2:
            a = np.linspace(-2, 2, 16)
            a = np.reshape(a, [16, 1])
            noise_code2 = np.ones((16, 1)) * a
        else:
            noise_code2 = np.ones((16, 1)) * code2
        print(noise_code2)

    gan = GAN(model_name="Test output")
    gan.plot_images(
        generator=generator,
        noise_input=np.random.uniform(
            -1.0,
            1.0,
            size=[16, latent_size],
        ),
        noise_label=noise_label,
        noise_codes=[noise_code1, noise_code2],
        show=True,
        step=step,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
