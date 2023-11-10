"""Main program for multilayer perceptron
"""
from dataclasses import dataclass, field
from typing import List, Optional

from keras import Model
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model, save_model
from keras.utils import plot_model
from numpy import ndarray


@dataclass
class MLP:
    """Main class for multilayer perceptron"""

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

    def add_layer(self, layer_type: str, units: int, input_dim: Optional[int]):
        """Add layer to model"""
        if layer_type == "Dense":
            if input_dim:
                self.model.add(Dense(units=units, input_dim=input_dim))
            else:
                self.model.add(Dense(units=units))

    def add_activation(self, activation: str):
        """Add activation to model"""
        self.model.add(Activation(activation=activation))

    def add_dropout(self, rate: float):
        """Add dropout"""
        self.model.add(Dropout(rate=rate))

    def summary(self):
        """Print out summary of the model"""
        self.model.summary()

    def plot_model(self, filepath: str = "data/mlp-mnist.png"):
        """Plot the model to a given file"""
        plot_model(self.model, to_file=filepath)

    def compile(self, loss: str, optimizer: str, metrics: List[str]):
        """Compile the model"""
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )

    def fit(self, data: ndarray, targets: ndarray, epochs: int, batchsize: int = 128):
        """Fit the training data"""
        self.model.fit(
            x=data,
            y=targets,
            batch_size=batchsize,
            epochs=epochs,
        )

    def evaluate(
        self,
        test_data: ndarray,
        test_targets: ndarray,
        batch_size: int = 128,
        verbose: str = "auto",
    ) -> float:
        """evaluate the model"""
        _, acc = self.model.evaluate(
            x=test_data,
            y=test_targets,
            verbose=verbose,
            batch_size=batch_size,
        )

        return acc

    def save(self):
        """Save keras model
        """
        if not self.model_path.endswith(".keras"):
            raise ValueError("Model name should end with keras")

        save_model(self.model, filepath=self.model_path)

    def predict(self, image: ndarray) -> ndarray:
        """predict image
        """
        result = self.model.predict(image)

        return result
