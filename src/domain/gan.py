"""Usecase for GANs"""
from dataclasses import dataclass
import math
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.models import Model


Models = Tuple[Model, Model, Model]


@dataclass
class GAN:
    """Base class for GANs"""

    kernel_size: int = 5
    batch_size: int = 64
    latent_size: int = 100
    train_steps: int = 40000
    model_name: str = "gan"
    save_intervals: int = 500
    image_size: int = 28

    def generator(
        self,
        inputs: Model,
        activation: Optional[str] = "sigmoid",
        labels: Optional[Model] = None,
    ) -> Model:
        """build generator"""
        image_resize = self.image_size // 4
        layer_filters = [128, 64, 32, 1]
        if labels is not None:
            inputs = [inputs, labels]
            x = layers.concatenate(inputs, axis=1)
        else:
            x = inputs

        x = layers.Dense(
            units=image_resize * image_resize * layer_filters[0],
        )(x)
        x = layers.Reshape(
            target_shape=(image_resize, image_resize, layer_filters[0]),
        )(x)

        for filters in layer_filters:
            if filters > layer_filters[-2]:
                strides = 2
            else:
                strides = 1

            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation="relu")(x)
            x = layers.Conv2DTranspose(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=strides,
                padding="same",
            )(x)

        if activation is not None:
            x = layers.Activation(activation=activation)(x)

        return Model(inputs=inputs, outputs=x, name="generator")

    def discriminator(
        self,
        inputs: Model,
        activation: Optional[str] = "sigmoid",
        labels: Optional[int] = None,
    ) -> Model:
        """build discriminator"""
        layer_filters = [32, 64, 128, 256]
        x = inputs

        for filters in layer_filters:
            if filters == layer_filters[-1]:
                strides = 1
            else:
                strides = 2

            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=strides,
                padding="same",
            )(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(units=1)(x)

        if activation is not None:
            print(activation)
            outputs = layers.Activation(activation=activation)(outputs)

        if labels is not None:
            y = layers.Dense(units=layer_filters[-2])(x)
            y = layers.Dense(units=labels)(y)
            y = layers.Activation(activation="softmax")(y)
            outputs = [outputs, y]

        return Model(inputs=inputs, outputs=outputs, name="discriminator")

    def train(self, models: Models, x_train: np.ndarray):
        """train gan"""
        generator, discriminator, adversarial = models

        for idx in range(self.train_steps):
            rand_indexex = np.random.randint(0, x_train.shape[0], size=self.batch_size)
            real_images = x_train[rand_indexex]
            noise = np.random.uniform(
                -1.0, 1.0, size=[self.batch_size, self.latent_size],
            )
            fake_images = generator.predict(noise)
            x = np.concatenate((real_images, fake_images))
            y = np.ones([2 * self.batch_size, 1])
            y[self.batch_size:, :] = 0.0
            score = discriminator.train_on_batch(x, y)
            log = f"{idx}: [discriminator loss: {score[0]}, acc: {score[1]}]"
            noise = np.random.uniform(
                -1.0, 1.0, size=[self.batch_size, self.latent_size],
            )
            y = np.ones([self.batch_size, 1])
            score = adversarial.train_on_batch(noise, y)
            log = f"{log}: [adversarial loss: {score[0]}, acc: {score[1]}]"
            print(log)

            if (idx + 1) % self.save_intervals == 0:
                self.plot_images(
                    generator=generator,
                    noise_input=np.random.uniform(
                        -1.0, 1.0, size=[16, self.latent_size],
                    ),
                    show=False,
                    step=(idx + 1),
                )

        generator.save(self.model_name + ".keras")

    def plot_images(
        self,
        generator: Model,
        noise_input: np.ndarray,
        noise_label: Optional[np.ndarray] = None,
        show: bool = False,
        step: int = 0,
    ):  # pylint: disable=too-many-arguments
        """plot images"""
        os.makedirs(self.model_name, exist_ok=True)
        filename = os.path.join(self.model_name, f"{step}.png")
        rows = int(math.sqrt(noise_input.shape[0]))

        if noise_label is not None:
            images = generator.predict([noise_input, noise_label])
        else:
            images = generator.predict(noise_input)

        plt.figure(figsize=(2.2, 2.2))
        num_images = images.shape[0]
        image_size = images.shape[1]

        for idx in range(num_images):
            plt.subplot(rows, rows, idx + 1)
            image = np.reshape(images[idx], [image_size, image_size])
            plt.imshow(image, cmap="gray")
            plt.axis("off")

        plt.savefig(filename)

        if show:
            plt.show()
        else:
            plt.close("all")

    def test_generator(self, generator: Model):
        """test the generator"""
        noise_input = np.random.uniform(
            -1.0, 1.0, size=[16, 100],
        )
        self.plot_images(
            generator=generator,
            noise_input=noise_input,
        )
