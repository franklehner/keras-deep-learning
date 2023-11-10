"""Usecase for mnist classification with cnn
"""
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import List, Optional, Tuple

from numpy import ndarray

from src.domain.models.convolutional_network import CNN, Layer

Metrics = List[str]
Filename = Optional[str]


@dataclass
class MnistClassifier:
    """Classifier"""

    layers: List[Layer]
    activations: List[Optional[str]]
    dropout: List[Optional[float]]
    max_pools: List[Tuple[int, int]]
    model_path: str
    batch_size: int
    cnn: CNN = field(init=False)

    def __post_init__(self):
        self.cnn = self.create_model()

    def create_model(self):
        """Create the CNN"""
        cnn = CNN(model_type="Sequential", model_path=self.model_path)
        for layer, activation, rate, pool_size in zip_longest(
            self.layers,
            self.activations,
            self.dropout,
            self.max_pools,
        ):
            cnn.add_layer(layer=layer)
            cnn.add_max_pooling(pool_size=pool_size)
            if layer.flatten:
                cnn.add_flatten()
            if activation:
                cnn.add_activation(activation=activation)
            if rate:
                cnn.add_dropout(rate=rate)
        cnn.summary()

        return cnn

    def compile(
        self, loss: str, optimizer: str, metrics: Metrics, filename: Filename = None,
    ):
        """Compile the model"""
        if filename:
            self.cnn.plot_model(filepath=filename)
        else:
            self.cnn.plot_model()

        self.cnn.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, x_train: ndarray, y_train: ndarray, epochs: int):
        """Train the model"""
        self.cnn.fit(
            data=x_train,
            targets=y_train,
            epochs=epochs,
            batch_size=self.batch_size,
        )

    def evaluate(self, test_data: ndarray, test_targets: ndarray) -> float:
        """Evaluate the model"""
        accuracy = self.cnn.evaluate(
            test_data=test_data,
            test_targets=test_targets,
            batch_size=self.batch_size,
        )

        return accuracy * 100

    def save(self):
        """Save the model"""
        self.cnn.save()
