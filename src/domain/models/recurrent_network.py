"""Entity of the recurrent network
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from keras import Model
from keras.layers import Activation, Dense, SimpleRNN
from keras.models import Sequential, load_model, save_model
from keras.utils import plot_model
from numpy import ndarray

InputShape = Tuple[int, int]


@dataclass
class Layer:
    """Layer"""

    layer_type: str
    units: int
    dropout: Optional[float] = None
    input_shape: Optional[InputShape] = None

    def __post_init__(self):
        if self.layer_type == "SimpleRNN":
            if not all([self.units, self.dropout, self.input_shape]):
                raise RuntimeError("Missing layer informations")


@dataclass
class RNN:
    """Recurrent network"""

    model_type: str
    model_path: str
    load: bool = False
    model: Model = field(init=False)

    def __post_init__(self):
        if self.load:
            self.model = load_model(self.model_path)
        else:
            if self.model_type == "Sequential":
                self.model = Sequential()

    def add_layer(self, layer: Layer):
        """Add a layer
        """
        if layer.layer_type == "SimpleRNN":
            self.model.add(
                SimpleRNN(
                    units=layer.units,
                    dropout=layer.dropout,
                    input_shape=layer.input_shape,
                ),
            )
        elif layer.layer_type == "Dense":
            self.model.add(Dense(units=layer.units))

    def add_activation(self, activation: str):
        """Add activation"""
        self.model.add(Activation(activation=activation))

    def summary(self):
        """Print out the summary of the model"""
        self.model.summary()

    def plot_model(self, filepath: str):
        """Plot model to a given filepath"""
        plot_model(model=self.model, to_file=filepath)

    def compile(self, loss: str, optimizer: str, metrics=List[str]):
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
        """Evaluate the model"""
        _, acc = self.model.evaluate(
            x=x_test,
            y=y_test,
            batch_size=batch_size,
            verbose="auto",
        )

        return acc

    def save(self):
        """Save as keras model"""
        save_model(model=self.model, filepath=self.model_path)

    def predict(self, image: ndarray) -> ndarray:
        """Predict the label of an image or images"""
        result = self.model.predict(x=image)

        return result
