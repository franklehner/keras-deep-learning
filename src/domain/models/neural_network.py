"""Entity for neural networks that are working with
Sequential
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from keras import Model
from keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    SimpleRNN,
)
from keras.models import Sequential, load_model, save_model
from keras.utils import plot_model
from numpy import ndarray

from src.domain.models.net_configurations import CNNShape, DenseShape, RNNShape


@dataclass
class DenseLayer:
    """Class with dense layers"""

    units: Optional[int] = None
    input_dim: Optional[DenseShape] = None
    layer: Dense = field(init=False)

    def __post_init__(self):
        if not self.input_dim:
            if self.units is not None:
                self.layer = Dense(units=self.units)
        else:
            if self.units is not None:
                self.layer = Dense(units=self.units, input_dim=self.input_dim)


@dataclass
class CNNLayer:
    """Class with conv2d layers"""

    kernel_size: int
    activation: str
    filters: int
    input_shape: Optional[CNNShape] = None
    layer: Conv2D = field(init=False)

    def __post_init__(self):
        if not self.input_shape:
            self.layer = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
            )
        else:
            self.layer = Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                input_shape=self.input_shape,
            )


@dataclass
class RNNLayer:
    """Class with simple rnn layer"""

    units: Optional[int] = None
    dropout: Optional[float] = None
    input_shape: Optional[RNNShape] = None
    layer: SimpleRNN = field(init=False)

    def __post_init__(self):
        if self.input_shape:
            if self.units is not None and self.dropout is not None:
                self.layer = SimpleRNN(
                    units=self.units,
                    dropout=self.dropout,
                    input_shape=self.input_shape,
                )
            else:
                raise RuntimeError("units or dropouts are None")
        else:
            if self.units is not None and self.dropout is not None:
                self.layer = SimpleRNN(
                    units=self.units,
                    dropout=self.dropout,
                )
            else:
                raise RuntimeError("units or dropout are None")


@dataclass
class NNSequential:
    """NeuronalNet"""

    path: str
    load: bool = False
    model: Model = field(init=False)

    def __post_init__(self):
        if self.load:
            self.model = load_model(self.path)
        else:
            self.model = Sequential()

    def add_layer(self, layer: Union[DenseLayer, RNNLayer, CNNLayer]):
        """Add a layer"""
        self.model.add(layer=layer.layer)

    def add_activation(self, activation: Optional[str]):
        """Add activation"""
        if activation is not None:
            self.model.add(Activation(activation=activation))

    def add_dropout(self, rate: Optional[float]):
        """add dropout layer"""
        if rate is not None:
            self.model.add(Dropout(rate=rate))

    def add_max_pooling(self, pool_size: Tuple[int, int]):
        """add maxpooling 2d"""
        self.model.add(MaxPooling2D(pool_size=pool_size))

    def flatten(self):
        """flatten the array"""
        self.model.add(Flatten())

    def save(self):
        """save model to file"""
        save_model(model=self.model, filepath=self.path)

    def plot_model(self, to_file: str):
        """plot model to file"""
        plot_model(model=self.model, to_file=to_file)

    def compile(self, loss: str, optimizer: str, metrics: List[str]):
        """compile the model"""
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )

    def fit(self, x_train: ndarray, y_train: ndarray, epochs: int, batch_size: int):
        """train the model"""
        self.model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
        )

    def evaluate(self, x_test: ndarray, y_test: ndarray, batch_size: int) -> float:
        """evaluate the model on the test data"""
        _, acc = self.model.evaluate(
            x=x_test,
            y=y_test,
            batch_size=batch_size,
        )

        return acc

    def summary(self):
        """Write summary"""
        self.model.summary()
