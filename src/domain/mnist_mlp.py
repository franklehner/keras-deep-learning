"""Usecase for mnist classification with mlp
"""
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import List, Optional, Tuple

from numpy import ndarray

from src.domain.models.multilayer_perceptron import MLP


@dataclass
class MnistClassifier:
    """MnistClassifier"""

    input_dim: int
    layers: List[Tuple[str, int]]
    activations: List[str]
    dropout: List[float]
    model_path: str
    batch_size: Optional[int] = None
    mlp: MLP = field(init=False)

    def __post_init__(self):
        self.mlp = self.create_model()

    def create_model(self):
        """create model"""
        mlp = MLP(model_type="Sequential", model_path=self.model_path)
        for layer, activation, rate in zip_longest(
            self.layers,
            self.activations,
            self.dropout,
        ):
            mlp.add_layer(layer_type=layer[0], units=layer[1], input_dim=self.input_dim)
            mlp.add_activation(activation=activation)
            if rate:
                mlp.add_dropout(rate=rate)
        mlp.summary()

        return mlp

    def compile(
        self,
        loss: str,
        optimizer: str,
        metrics: List[str],
        filename: Optional[str] = None,
    ):
        """Compile the model"""
        if filename:
            self.mlp.plot_model(filepath=filename)
        else:
            self.mlp.plot_model()

        self.mlp.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x_train: ndarray, y_train: ndarray, epochs: int):
        """fit the model"""
        if not self.batch_size:
            self.batch_size = 128

        self.mlp.fit(
            data=x_train,
            targets=y_train,
            epochs=epochs,
            batchsize=self.batch_size,
        )

    def evaluate(self, test_data: ndarray, test_targets: ndarray) -> float:
        """Evaluate the test data
        """
        if not self.batch_size:
            self.batch_size = 128
        accuracy = self.mlp.evaluate(
            test_data=test_data,
            test_targets=test_targets,
            batch_size=self.batch_size,
        )

        return accuracy * 100.0

    def save(self):
        """Save the model"""
        self.mlp.save()
