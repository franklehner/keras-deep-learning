"""Dense net model"""
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
    concatenate,
)
from keras.models import Model


@dataclass
class DenseNet:
    """Fully connected network"""

    depth: int
    growth_rate: int
    num_dense_blocks: int
    input_shape: Tuple[int, ...]
    num_classes: int
    compression_rate: float
    model_type: str = field(init=False)

    def __post_init__(self):
        self.model_type = f"DenseNet{self.depth}_"

    def dense_blocks_layer(self, x: Model, num_bottleneck_layers: int) -> Model:
        """generate dense blocks"""
        for _ in range(num_bottleneck_layers):
            y = BatchNormalization()(x)
            y = Activation("relu")(y)
            y = Conv2D(
                filters=4 * self.growth_rate,
                kernel_size=1,
                padding="same",
                kernel_initializer="he_normal",
            )(y)
            y = BatchNormalization()(y)
            y = Activation("relu")(y)
            y = Conv2D(
                filters=self.growth_rate,
                kernel_size=3,
                padding="same",
                kernel_initializer="he_normal",
            )(y)
            x = concatenate([x, y])

        return x

    def initial_layer(self, inputs: Input, num_filters: int) -> Model:
        """first layer"""
        x = BatchNormalization()(inputs)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal",
        )(x)

        x = concatenate([inputs, x])

        return x

    def generate_dense_blocks(self, x: Model, num_filters: int) -> Model:
        """generate dense blocks"""
        num_bottleneck_layers = (self.depth - 4) // (2 * self.num_dense_blocks)
        for i in range(self.num_dense_blocks):
            x = self.dense_blocks_layer(x=x, num_bottleneck_layers=num_bottleneck_layers)

            if i == self.num_dense_blocks - 1:
                continue

            num_filters += num_bottleneck_layers * self.growth_rate
            num_filters = int(num_filters * self.compression_rate)

            y = BatchNormalization()(x)
            y = Conv2D(
                filters=num_filters,
                kernel_size=1,
                padding="same",
                kernel_initializer="he_normal",
            )(y)
            x = AveragePooling2D()(y)

        x = AveragePooling2D(pool_size=(8, 8))(x)
        y = Flatten()(x)
        outputs = Dense(
            self.num_classes,
            kernel_initializer="he_normal",
            activation="softmax",
        )(y)

        return outputs

    def get_num_filters(self) -> int:
        """get number of filters"""
        return 2 * self.growth_rate

    def train_net(self) -> Model:
        """train the model"""
        num_filters = self.get_num_filters()
        inputs = Input(shape=self.input_shape)

        x = self.initial_layer(inputs=inputs, num_filters=num_filters)
        outputs = self.generate_dense_blocks(x=x, num_filters=num_filters)

        model = Model(inputs=inputs, outputs=outputs)

        return model
