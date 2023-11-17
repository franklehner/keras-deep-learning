"""Resnet
"""
import os
from dataclasses import dataclass, field
from typing import Tuple

from keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    add,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model


@dataclass
class ResNet:
    """residual network"""

    version: int
    depth: int
    input_shape: Tuple[int, ...]
    num_classes: int
    model_type: str = field(init=False)

    def __post_init__(self):
        if self.version == 2:
            if (self.depth - 2) % 9 != 0:
                raise ValueError("Depth must be 9n + 2")
        elif self.version == 1:
            if (self.depth - 2) % 6 != 0:
                raise ValueError("Depth must be 6n + 2")
        else:
            raise ValueError("Version must be 1 or 2")

        self.model_type = f"ResNet{self.depth}v{self.version}"

    def resnet_layer(
        self,
        inputs: Model,
        num_filters: int = 16,
        kernel_size: int = 3,
        strides: int = 1,
        activation: str = "relu",
        batch_normalization: bool = True,
        conv_first: bool = True,
    ) -> Model:  # pylint: disable=too-many-arguments
        """layer of resnet"""
        conv = Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
        )
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation=activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation=activation)(x)
            x = conv(x)

        return x

    def resnet_v1(self):
        """Train resnet version 1"""
        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)
        inputs = Input(shape=self.input_shape)

        x = self.resnet_layer(inputs=inputs)

        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2

                y = self.resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    strides=strides,
                )
                y = self.resnet_layer(
                    inputs=y,
                    num_filters=num_filters,
                    activation=None,
                )

                if stack > 0 and res_block == 0:
                    x = self.resnet_layer(
                        inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )

                x = add([x, y])
                x = Activation("relu")(x)

            num_filters *= 2

        x = AveragePooling2D(pool_size=(8, 8))(x)
        y = Flatten()(x)
        outputs = Dense(
            self.num_classes,
            activation="softmax",
            kernel_initializer="he_normal",
        )(y)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def resnet_v2(self):
        """train resnet version 2"""
        num_filters_in = 16
        num_res_blocks = int((self.depth - 2) / 9)
        inputs = Input(shape=self.input_shape)

        x = self.resnet_layer(
            inputs=inputs,
            num_filters=num_filters_in,
            conv_first=True,
        )

        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = "relu"
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:
                        strides = 2

                y = self.resnet_layer(
                    inputs=x,
                    kernel_size=1,
                    strides=strides,
                    activation=activation,
                    batch_normalization=batch_normalization,
                    conv_first=False,
                )
                y = self.resnet_layer(
                    inputs=y,
                    num_filters=num_filters_in,
                    conv_first=False,
                )

                y = self.resnet_layer(
                    inputs=y,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    conv_first=False,
                )

                if res_block == 0:
                    x = self.resnet_layer(
                        inputs=x,
                        num_filters=num_filters_out,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )

                x = add([x, y])

            num_filters_in = num_filters_out

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = AveragePooling2D(pool_size=(8, 8))(x)
        y = Flatten()(x)
        outputs = Dense(
            self.num_classes,
            activation="softmax",
            kernel_initializer="he_normal",
        )(y)

        model = Model(inputs=inputs, outputs=outputs)

        return model


@dataclass
class NeuralNetwork:
    """Abstract"""

    model: Model
    model_type: str
    dataset_name: str
    save_dir: str = "saved_models"
    name: str = field(init=False)
    filepath: str = field(init=False)

    def __post_init__(self):
        self.name = "%s_%s_model.{epoch:03d}.hd5" % (self.dataset_name, self.model_type)
        self.filepath = os.path.join(self.save_dir, self.name)

    def lr_scheduler(self, epoch: int):
        """lr_scheduler"""
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1

        print(f"Learning rage: {lr}")

        return lr

    def compile(self):
        """Compile the model"""
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(
                learning_rate=self.lr_scheduler(0),
            ),
            metrics=["acc"],
        )

    def summary(self):
        """Summary of the model"""
        self.model.summary()

    def plot_model(self, to_file: str):
        """plot model"""
        plot_model(model=self.model, to_file=to_file, show_shapes=True)
