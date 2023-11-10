"""Entity of the convolusional network
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from keras import Model
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model, save_model
from keras.utils import plot_model
from numpy import ndarray


@dataclass
class Layer:
    """Layer"""

    layer_type: str
    kernel_size: Optional[int] = None
    activation: Optional[str] = None
    filters: Optional[int] = None
    input_shape: Optional[Tuple[int, int, int]] = None
    units: Optional[int] = None
    flatten: bool = False

    def __post_init__(self):
        if self.layer_type == "Conv2D":
            if not all([self.kernel_size, self.activation, self.filters]):
                raise RuntimeError("Missing layer informations")
        elif self.layer_type == "Dense":
            if not self.units:
                raise RuntimeError("Missing layer informations")


@dataclass
class CNN:
    """Convolutional network"""

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
        """Add a layer"""
        if layer.layer_type == "Conv2D":
            if layer.input_shape:
                conv2d_layer = Conv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    activation=layer.activation,
                    input_shape=layer.input_shape,
                )
            else:
                conv2d_layer = Conv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    activation=layer.activation,
                )
            self.model.add(conv2d_layer)
        elif layer.layer_type == "Dense":
            self.model.add(Dense(units=layer.units))
        else:
            raise RuntimeError("Wrong layer")

    def add_activation(self, activation: Optional[str]):
        """add activation to the model"""
        if activation is not None:
            self.model.add(Activation(activation=activation))

    def add_max_pooling(self, pool_size: int):
        """add pool size"""
        if pool_size:
            self.model.add(MaxPooling2D(pool_size=pool_size))

    def add_flatten(self):
        """flatten the array"""
        self.model.add(Flatten())

    def add_dropout(self, rate: Optional[float]):
        """Add dropout rate to the model"""
        if rate is not None:
            self.model.add(Dropout(rate=rate))

    def summary(self):
        """Print out the summary of the model"""
        self.model.summary()

    def plot_model(self, filepath: str = "data/cnn-mnist.png"):
        """Plot model to a given file"""
        plot_model(model=self.model, to_file=filepath)

    def compile(self, loss: str, optimizer: str, metrics: List[str]):
        """Compile the model"""
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )

    def fit(self, data: ndarray, targets: ndarray, epochs: int, batch_size: int):
        """Train the model with the training data"""
        self.model.fit(
            x=data,
            y=targets,
            epochs=epochs,
            batch_size=batch_size,
        )

    def evaluate(
        self,
        test_data: ndarray,
        test_targets: ndarray,
        batch_size: int,
        verbose: str = "auto",
    ):
        """evaluate the model on the test data"""
        _, acc = self.model.evaluate(
            x=test_data,
            y=test_targets,
            verbose=verbose,
            batch_size=batch_size,
        )

        return acc

    def save(self):
        """Save the keras model"""
        if not self.model_path.endswith(".keras"):
            raise ValueError("Model name should end with .keras")

        save_model(model=self.model, filepath=self.model_path)

    def predict(self, image: ndarray) -> ndarray:
        """Predict the label of the image"""
        result = self.model.predict(image)

        return result
