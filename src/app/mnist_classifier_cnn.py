"""Application for mnist classifier cnn
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypedDict

from numpy import ndarray

from src.domain.mnist_cnn import MnistClassifier
from src.domain.models.convolutional_network import Layer
from src.infra.datasets import Mnist

InputShape = Tuple[int, int, int]


class Conv2DParams(TypedDict):
    """Parameter for conv2d layers"""

    kernel_size: int
    filters: int
    activation: str
    input_shape: InputShape


def load_mnist() -> Mnist:
    """load data"""
    mnist = Mnist()

    return mnist


def generate_conv2d_layers(count: int, layer_params: Conv2DParams) -> List[Layer]:
    """generate layers for conv2d"""
    layers = []
    first_layer = Layer(
        layer_type="Conv2D",
        kernel_size=layer_params["kernel_size"],
        filters=layer_params["filters"],
        activation=layer_params["activation"],
        input_shape=layer_params["input_shape"],
    )
    layers.append(first_layer)
    for _ in range(count - 1):
        layers.append(
            Layer(
                layer_type="Conv2D",
                kernel_size=layer_params["kernel_size"],
                activation=layer_params["activation"],
                filters=layer_params["filters"],
            ),
        )

    layers[-1].flatten = True

    return layers


def generate_dense_layers(count: int, units: int) -> List[Layer]:
    """generate layers"""
    layers = [
        Layer(
            layer_type="Dense",
            units=units,
        )
        for _ in range(count)
    ]

    return layers


def generate_activations(
    count: int,
    last: str,
    activation: Optional[str] = None,
) -> List[Optional[str]]:
    """Generate activations"""
    activations = [activation for _ in range(count)]
    activations[-1] = last

    return activations


def generte_dropouts(
    count: int,
    last_rate: float,
    rate: Optional[float] = None,
) -> List[Optional[float]]:
    """generate dropouts"""
    dropouts = [rate for _ in range(count)]
    dropouts[-2] = last_rate

    return dropouts


def generate_maxpool(count: int, pool_size: int) -> List[Tuple[int, int]]:
    """genrate maxpool"""
    max_pools = [(pool_size, pool_size) for _ in range(count)]

    return max_pools


@dataclass
class ModelTrainer:
    """Train and save model"""

    layers: List[Layer]
    activations: List[Optional[str]]
    dropouts: List[Optional[float]]
    max_pools: List[Tuple[int, int]]
    model_path: str
    batchsize: int
    model: MnistClassifier = field(init=False)

    def __post_init__(self):
        self.model = MnistClassifier(
            layers=self.layers,
            activations=self.activations,
            dropout=self.dropouts,
            max_pools=self.max_pools,
            model_path=self.model_path,
            batch_size=self.batchsize,
        )

    def compile(self):
        """compile model"""
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

    def fit(self, x_train: ndarray, y_train: ndarray, epochs: int):
        """train model"""
        self.model.fit(
            x_train=x_train,
            y_train=y_train,
            epochs=epochs,
        )

    def evaluate(self, x_test: ndarray, y_test: ndarray) -> float:
        """evaluate model"""
        accuracy = self.model.evaluate(
            test_data=x_test,
            test_targets=y_test,
        )

        return accuracy

    def save(self):
        """save model"""
        self.model.save()
