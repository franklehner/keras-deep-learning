"""conditional gan"""
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import layers, optimizers
from keras.models import Model, load_model
from keras.utils import to_categorical
from numpy import ndarray

from src.app.dataset_reader import MnistDataSet
from src.domain.models.datasets import DataSet


Models = Tuple[Model, Model, Model]
Data = Tuple[ndarray, ndarray]


@dataclass
class CGAN:
    """Main class for image generator"""

    kernel_size: int = 5
    save_interval: int = 500
    batch_size: int = 64
    latent_size: int = 100
    train_steps: int = 40000
    model_name: str = "cgan_mnist"

    def get_dataset(self) -> DataSet:
        """get mnist dataset"""
        dataset = MnistDataSet()

        return dataset.load_dataset()

    def build_generator(self, inputs: Model, labels: Model, image_size: int) -> Model:
        """build generator"""
        image_resize = image_size // 4
        layer_filters: List[int] = [128, 64, 32, 1]

        x = layers.concatenate([inputs, labels], axis=1)
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

        x = layers.Activation(activation="sigmoid")(x)
        generator = Model(
            inputs=[inputs, labels],
            outputs=x,
            name="generator",
        )

        return generator

    def build_discriminator(
        self, inputs: Model, labels: Model, image_size: int,
    ) -> Model:
        """build discriminator"""
        layer_filters: List[int] = [32, 64, 128, 256]

        x = inputs
        y = layers.Dense(
            units=image_size * image_size,
        )(labels)
        y = layers.Reshape(
            target_shape=(image_size, image_size, 1),
        )(y)
        x = layers.concatenate([x, y])

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
        x = layers.Dense(units=1)(x)
        x = layers.Activation(activation="sigmoid")(x)
        discriminator = Model(
            inputs=[inputs, labels],
            outputs=x,
            name="discriminator",
        )

        return discriminator

    def build_and_train_models(self):
        """build and train the models"""
        mnist_data = self.get_dataset()
        x_train = mnist_data.x_train
        y_train = mnist_data.y_train
        x_train = np.reshape(x_train, [-1, x_train.shape[1], x_train.shape[1], 1])
        x_train = x_train.astype("float32") / 255
        num_labels = np.amax(y_train) + 1
        y_train = to_categorical(y_train)
        lr = 2e-4
        input_shape = (x_train.shape[1], x_train.shape[1], 1)
        label_shape = (num_labels,)
        inputs = layers.Input(shape=input_shape, name="discriminator_input")
        labels = layers.Input(shape=label_shape, name="class_labels")
        discriminator = self.build_discriminator(
            inputs=inputs,
            labels=labels,
            image_size=x_train.shape[1],
        )
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
        inputs = layers.Input(
            shape=input_shape,
            name="z_input",
        )
        generator = self.build_generator(
            inputs=inputs,
            labels=labels,
            image_size=x_train.shape[1],
        )
        generator.summary()
        optimizer = optimizers.RMSprop(
            learning_rate=lr * 0.5,
        )
        discriminator.trainable = False
        outputs = discriminator([generator([inputs, labels]), labels])
        adversarial = Model(
            inputs=[inputs, labels],
            outputs=outputs,
            name=self.model_name,
        )
        adversarial.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        adversarial.summary()
        self.train(
            models=(generator, discriminator, adversarial),
            data=(x_train, y_train),
            num_labels=num_labels,
        )

    def train(self, models: Models, data: Data, num_labels: int):  # pylint: disable=too-many-locals
        """train the models"""
        generator, discriminator, adversarial = models
        x_train, y_train = data
        noise_input = np.random.uniform(
            -1.0,
            1.0,
            size=[16, self.latent_size],
        )
        noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]
        print(
            self.model_name,
            "Labels for genrated images",
            np.argmax(noise_class, axis=1),
        )

        for i in range(self.train_steps):
            rand_indexes = np.random.randint(
                0,
                x_train.shape[0],
                size=self.batch_size,
            )
            real_images = x_train[rand_indexes]
            real_labels = y_train[rand_indexes]
            noise = np.random.uniform(
                -1.0,
                1.0,
                size=[self.batch_size, self.latent_size],
            )
            fake_labels = np.eye(
                num_labels,
            )[np.random.choice(num_labels, self.batch_size)]
            fake_images = generator.predict(
                [noise, fake_labels],
            )
            x = np.concatenate((real_images, fake_images))
            labels = np.concatenate((real_labels, fake_labels))
            y = np.ones([2 * self.batch_size, 1])
            y[self.batch_size:, :] = 0.0

            loss, acc = discriminator.train_on_batch([x, labels], y)
            log = f"{i}: [discriminator loss: {loss}, acc: {acc}]"
            noise = np.random.uniform(
                -1.0,
                1.0,
                size=[self.batch_size, self.latent_size],
            )
            fake_labels = np.eye(
                num_labels,
            )[np.random.choice(num_labels, self.batch_size)]
            y = np.ones([self.batch_size, 1])
            loss, acc = adversarial.train_on_batch([noise, fake_labels], y)
            log = f"{log} [adversarial loss: {loss}, acc: {acc}]"
            print(log)

            if (i + 1) % self.save_interval == 0:
                self.plot_images(
                    generator=generator,
                    noise_input=noise_input,
                    noise_class=noise_class,
                    step=(i + 1),
                    model_name=self.model_name,
                )

        generator.save(self.model_name + ".h5")

    def plot_images(
        self,
        generator: Model,
        noise_input: ndarray,
        noise_class: ndarray,
        show: bool = False,
        step: int = 0,
        model_name: str = "cgan",
    ):
        """plot images"""
        os.makedirs(model_name, exist_ok=True)
        filename = os.path.join(model_name, f"{step}.png")
        images = generator.predict([noise_input, noise_class])
        print(
            model_name,
            "labels for generated images:",
            np.argmax(noise_class, axis=1),
        )
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

    def test_generator(self, generator: Model, class_label: Optional[int] = None):
        """test the generator"""
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        step = 0
        if class_label is None:
            num_labels = 10
            noise_class = np.eye(
                num_labels,
            )[np.random.choice(num_labels, 16)]
        else:
            noise_class = np.zeros((16, 10))
            noise_class[:, class_label] = 1
            step = class_label

        self.plot_images(
            generator=generator,
            noise_input=noise_input,
            noise_class=noise_class,
            show=True,
            step=step,
            model_name="test_outputs",
        )

    def load_model(self, filepath: str) -> Model:
        """load model"""
        return load_model(filepath=filepath)
