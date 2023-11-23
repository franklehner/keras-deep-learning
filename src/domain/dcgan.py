"""Generate images via DCGAN"""
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers
from keras.models import Model, load_model
from numpy import ndarray

from src.app.dataset_reader import MnistDataSet
from src.domain.models.datasets import DataSet

Models = Tuple[Model, Model, Model]


@dataclass
class DCGAN:
    """Main class for image generator"""

    kernel_size: int = 5
    save_interval: int = 500
    batch_size: int = 64
    latent_size: int = 100
    train_steps: int = 40000

    def get_dataset(self) -> DataSet:
        """get mnist dataset"""
        dataset = MnistDataSet()

        return dataset.load_dataset()

    def build_generator(self, inputs: Model, image_size: int) -> Model:
        """build generator"""
        image_resize = image_size // 4
        layer_filters: List[int] = [128, 64, 32, 1]
        x = layers.Dense(
            units=image_resize * image_resize * layer_filters[0],
        )(inputs)
        x = layers.Reshape(
            (image_resize, image_resize, layer_filters[0]),
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

        x = layers.Activation(activation="sigmoid")(x)
        generator = Model(inputs=inputs, outputs=x, name="generator")

        return generator

    def build_discriminator(self, inputs: Model) -> Model:
        """build the discriminator"""
        layer_filters: List[int] = [32, 64, 128, 256]
        x = inputs
        strides = 2

        for filters in layer_filters:
            if filters == layer_filters[-1]:
                strides = 1

            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=strides,
                padding="same",
            )(x)

        x = layers.Flatten()(x)
        x = layers.Dense(units=1)(x)
        x = layers.Activation(activation="sigmoid")(x)
        discriminator = Model(inputs=inputs, outputs=x, name="discriminator")

        return discriminator

    def build_and_train_models(self):
        """build and train models"""
        mnist_data = self.get_dataset()
        x_train = mnist_data.x_train
        image_size = x_train.shape[1]
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_train = x_train.astype("float32") / 255
        model_name = "dcgan_mnist"
        lr = 2e-4
        input_shape = (image_size, image_size, 1)
        inputs = layers.Input(shape=input_shape, name="discriminator_input")
        discriminator = self.build_discriminator(inputs=inputs)
        optimizer = optimizers.RMSprop(
            learning_rate=lr,
        )
        discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        discriminator.summary()
        input_shape = (self.latent_size,)
        inputs = layers.Input(shape=input_shape, name="z_input")
        generator = self.build_generator(inputs=inputs, image_size=image_size)
        generator.summary()
        optimizer = optimizers.RMSprop(
            learning_rate=lr * 0.5,
        )
        discriminator.trainable = False
        adversarial = Model(
            inputs=inputs,
            outputs=discriminator(generator(inputs)),
            name=model_name,
        )
        adversarial.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        models = (generator, discriminator, adversarial)
        self.train(models=models, x_train=x_train, model_name=model_name)

    def train(self, models: Models, x_train: ndarray, model_name: str):
        """train the models"""
        generator, discriminator, adversarial = models
        noise_input = np.random.uniform(
            -1.0,
            1.0,
            size=[16, self.latent_size],
        )

        for i in range(self.train_steps):
            rand_indexes = np.random.randint(
                0,
                x_train.shape[0],
                size=self.batch_size,
            )
            real_images = x_train[rand_indexes]
            noise = np.random.uniform(
                -1.0,
                1.0,
                size=[self.batch_size, self.latent_size],
            )
            fake_images = generator.predict(noise)
            x = np.concatenate((real_images, fake_images))
            y = np.ones([2 * self.batch_size, 1])
            y[self.batch_size:, :] = 0.0
            loss, acc = discriminator.train_on_batch(x, y)
            log = f"{i}: [discriminator loss: {loss}, acc: {acc}]"
            noise = np.random.uniform(
                -1.0,
                1.0,
                size=[self.batch_size, self.latent_size],
            )
            y = np.ones([self.batch_size, 1])
            loss, acc = adversarial.train_on_batch(noise, y)
            log = f"{log} [adversarial loss: {loss}, acc: {acc}]"
            print(log)

            if (i + 1) % self.save_interval == 0:
                self.plot_images(
                    generator=generator,
                    noise_input=noise_input,
                    show=False,
                    step=(i + 1),
                    model_name=model_name,
                )

        generator.save(model_name + ".h5")

    def plot_images(
        self,
        generator: Model,
        noise_input: ndarray,
        show: bool = False,
        step: int = 0,
        model_name: str = "dcgan_mnist",
    ):  # pylint: disable=too-many-arguments
        """plot images"""
        os.makedirs(model_name, exist_ok=True)
        filename = os.path.join(model_name, f"{step}.png")
        images = generator.predict(noise_input)
        plt.figure(figsize=(2.2, 2.2))
        num_images = images.shape[0]
        image_size = images.shape[1]
        rows = int(math.sqrt(noise_input.shape[0]))

        for i in range(num_images):
            plt.subplot(rows, rows, i + 1)
            image = np.reshape(images[i], [image_size, image_size])
            plt.imshow(image, cmap="gray")
            plt.axis("off")

        plt.savefig(filename)
        if show:
            plt.show()
        else:
            plt.close("all")

    def test_generator(self, generator: Model):
        """test the generator"""
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        self.plot_images(
            generator=generator,
            noise_input=noise_input,
            show=True,
            model_name="test_outputs",
        )

    def load_model(self, filepath: str) -> Model:
        """load a model"""
        return load_model(filepath=filepath)
