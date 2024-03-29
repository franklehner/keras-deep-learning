"""Entity for neural networks that are working with
Sequential
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import keras.backend as k
from keras import Model
from keras.layers import (
    Activation,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Reshape,
    SimpleRNN,
    concatenate,
)
from keras.models import Sequential, load_model, save_model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from numpy import ndarray

from src.domain.models.net_configurations import DenseShape, InputShape


@dataclass
class ShapeLayer:
    """shape_layer"""

    int_shape: Optional[int] = 0

    def get_shape(self, model: Model) -> Tuple[int, ...]:
        """get shape"""
        shape = k.int_shape(model)

        return shape[1:]


@dataclass
class DenseLayer:
    """Class with dense layers"""

    units: Optional[int] = None
    input_dim: Optional[DenseShape] = None
    activation: Optional[str] = None
    name: Optional[str] = None
    layer: Dense = field(init=False)

    def __post_init__(self):
        params = {
            k: v
            for k, v in zip(
                ["units", "activation", "input_dim"],
                [self.units, self.activation, self.input_dim],
            )
            if v is not None
        }
        self.layer = Dense(**params)


@dataclass
class CNNLayer:  # pylint: disable=too-many-instance-attributes
    """Class with conv2d layers"""

    kernel_size: Optional[int] = None
    activation: Optional[str] = None
    filters: Optional[int] = None
    input_shape: Optional[InputShape] = None
    dilation_rate: Optional[float] = None
    padding: Optional[str] = None
    strides: Optional[str] = None
    layer: Conv2D = field(init=False)

    def __post_init__(self):
        params = {
            k: v
            for k, v in zip(
                [
                    "kernel_size",
                    "activation",
                    "filters",
                    "input_shape",
                    "dilation_rate",
                    "padding",
                    "strides",
                ],
                [
                    self.kernel_size,
                    self.activation,
                    self.filters,
                    self.input_shape,
                    self.dilation_rate,
                    self.padding,
                    self.strides,
                ],
            )
            if v is not None
        }

        self.layer = Conv2D(**params)


@dataclass
class CNNTransposeLayer:  # pylint: disable=too-many-instance-attributes
    """Class with conv2d layers"""

    kernel_size: Optional[int] = None
    activation: Optional[str] = None
    filters: Optional[int] = None
    input_shape: Optional[InputShape] = None
    dilation_rate: Optional[float] = None
    padding: Optional[str] = None
    strides: Optional[str] = None
    name: Optional[str] = None
    layer_name: Optional[str] = None
    layer: Conv2D = field(init=False)

    def __post_init__(self):
        params = {
            k: v
            for k, v in zip(
                [
                    "kernel_size",
                    "activation",
                    "filters",
                    "input_shape",
                    "dilation_rate",
                    "padding",
                    "strides",
                    "name",
                ],
                [
                    self.kernel_size,
                    self.activation,
                    self.filters,
                    self.input_shape,
                    self.dilation_rate,
                    self.padding,
                    self.strides,
                    self.name,
                ],
            )
            if v is not None
        }

        self.layer = Conv2DTranspose(**params)


@dataclass
class RNNLayer:
    """Class with simple rnn layer"""

    units: Optional[int] = None
    dropout: Optional[float] = None
    activation: Optional[str] = None
    input_shape: Optional[InputShape] = None
    layer: SimpleRNN = field(init=False)

    def __post_init__(self):
        params = {
            k: v
            for k, v in zip(
                ["units", "dropout", "activation", "input_shape"],
                [self.units, self.dropout, self.activation, self.input_shape],
            )
            if v is not None
        }
        if self.input_shape:
            if len(self.input_shape) > 2:
                self.input_shape = self.input_shape[:2]
                params["input_shape"] = self.input_shape

        self.layer = SimpleRNN(**params)


@dataclass
class ActivationLayer:
    """Layer for activations"""

    activation: Optional[str] = None
    layer: Activation = field(init=False)

    def __post_init__(self):
        if self.activation is not None:
            self.layer = Activation(activation=self.activation)


@dataclass
class DropoutLayer:
    """Layer for dropout"""

    rate: Optional[float] = None
    layer: Dropout = field(init=False)

    def __post_init__(self):
        if self.rate is not None:
            self.layer = Dropout(rate=self.rate)


@dataclass
class FlattenLayer:
    """Flatten the net"""

    flatten: Optional[str] = None
    layer: Flatten = field(init=False)

    def __post_init__(self):
        self.layer = Flatten()


@dataclass
class MaxPoolingLayer:
    """maxpooling"""

    pool_size: Optional[Tuple[int, int]] = None
    layer: MaxPooling2D = field(init=False)

    def __post_init__(self):
        if self.pool_size is None:
            self.layer = MaxPooling2D()
        else:
            self.layer = MaxPooling2D(pool_size=self.pool_size)


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

    def save(self, filepath: str) -> None:
        """save model to file"""
        filepath = self.path
        save_model(model=self.model, filepath=filepath)

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

    def fit(
        self,
        x_train: ndarray,
        y_train: ndarray,
        epochs: int,
        batch_size: int,
        validation_data: Optional[Tuple[ndarray, ndarray]] = None,
    ):  # pylint: disable=too-many-arguments
        """train the model"""
        if validation_data is not None:
            self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
            )
        else:
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

    def predict(self, images: ndarray) -> ndarray:
        """predict images"""
        return self.model.predict(x=images)

    def summary(self):
        """Write summary"""
        self.model.summary()


@dataclass
class NNFunctional:
    """functional api"""

    path: str
    load: bool = False
    model: Model = field(init=False)

    def __post_init__(self):
        if self.load:
            self.model = load_model(filepath=self.path)

    def model_input(self, input_shape: Optional[InputShape]) -> Model:
        """InputLayer"""
        if input_shape is not None:
            inputs = Input(shape=input_shape)

        return inputs

    def add_layer(
        self,
        model: Model,
        layer: Union[DenseLayer, RNNLayer, CNNLayer, CNNTransposeLayer],
    ) -> Model:
        """add layer to model"""
        model = layer.layer(model)

        return model

    def reshape(self, model: Model, input_shape: InputShape) -> Model:
        """reshape layer"""
        model = Reshape(input_shape)(model)

        return model

    def get_shape(self, model: Model, layer: ShapeLayer) -> InputShape:
        """get shape of the model"""
        shape = layer.get_shape(model=model)

        return shape

    def generate_model(self, inputs: Model, outputs: Model) -> Model:
        """generate model from in- and output"""
        return Model(inputs=inputs, outputs=outputs)

    def predict(self, images: ndarray) -> ndarray:
        """predict images"""
        return self.model.predict(x=images)

    def summary(self) -> None:
        """write summary of the model"""

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

    def fit(
        self,
        x: ndarray,
        y: ndarray,
        validation_data: List[ndarray],
        epochs: int,
        batch_size: int,
    ):  # pylint: disable=too-many-arguments
        """train the model"""
        self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
        )

    def concatenate(self, inputs: List[Model]) -> "NNFunctional":
        """concatenate"""
        self.model = concatenate(inputs=inputs)

        return self

    def evaluate(
        self,
        x: ndarray,
        y: ndarray,
        batch_size: int,
    ) -> Tuple[List[float], float]:
        """evaluate the model on the test data"""
        score = self.model.evaluate(
            x=x,
            y=y,
            batch_size=batch_size,
        )

        return score

    def save(self, filepath: str) -> None:
        """Save model"""
        save_model(self.model, filepath=filepath)


@dataclass
class Concatenation:
    """Concatenate nets"""

    inputs: List[Model]

    def concatenate(self) -> Model:
        """concatenation"""

        return concatenate(inputs=self.inputs)


@dataclass
class NeuralNet:
    """abstract network"""

    model: Model
    model_type: str
    dataset_name: str
    save_dir: str = "saved_models"
    name: str = field(init=False)
    filepath: str = field(init=False)

    def __post_init__(self):
        self.name = "%s_%s_model.{epoch:03d}.hd5" % (
            self.dataset_name,
            self.model_type,
        )
        self.filepath = os.path.join(self.save_dir, self.name)

    def lr_schedule(self, epoch: int) -> float:
        """lr_scheduler"""
        lr: float = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1

        print(f"Learning rate: {lr}")

        return lr

    def compile(self):
        """compile the model"""
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=RMSprop(
                learning_rate=self.lr_schedule(0),
            ),
            metrics=["acc"],
        )

    def summary(self):
        """summary of the model"""
        self.model.summary()

    def plot_model(self, to_file: str):
        """plot model"""
        plot_model(model=self.model, to_file=to_file, show_shapes=True)
