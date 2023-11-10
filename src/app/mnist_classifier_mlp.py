"""Application for mnist classifier mlp
"""
from typing import List, Tuple
from numpy import ndarray
import matplotlib.pyplot as plt

from src.domain.mnist_mlp import MnistClassifier, MnistLoader
from src.domain.models.multilayer_perceptron import MLP
from src.infra.datasets import Mnist as MnistDatasets


def load_mnist() -> MnistDatasets:
    """load data"""
    mnist = MnistDatasets()

    return mnist


def generate_layers(
    count: int, layer: str, units: int, num_labels: int
) -> List[Tuple[str, int]]:
    """generate layers"""
    layers = [(layer, units) for _ in range(count - 1)]
    layers.append((layer, num_labels))

    return layers


def generate_activations(count: int, activation: str, last: str) -> List[str]:
    """generate activations"""
    activations = [activation for _ in range(count - 1)]
    activations.append(last)

    return activations


def generate_dropouts(count: int, rate: float) -> List[float]:
    """generate dropouts"""
    dropouts = [rate for _ in range(count)]

    return dropouts


def run(  # pylint: disable=too-many-arguments
    mnist: MnistDatasets,
    layers: List[Tuple[str, int]],
    activations: List[str],
    dropouts: List[float],
    model_path: str,
    batchsize: int,
    epochs: int,
):
    """runner method"""
    model = MnistClassifier(
        input_dim=mnist.input_size,
        layers=layers,
        activations=activations,
        dropout=dropouts,
        model_path=model_path,
        batch_size=batchsize,
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.fit(mnist.x_train, mnist.y_train, epochs=epochs)
    accuracy = model.evaluate(
        test_data=mnist.x_test,
        test_targets=mnist.y_test,
    )
    model.save()

    print(f"\n\nTest accuracy: {accuracy}")


def plot_image(image: ndarray, cmap: str = "gray"):
    """Plot image
    """
    plt.imshow(image, cmap=cmap)
    plt.show()


def load_model(filepath: str) -> MLP:
    """Load trained model
    """
    model = MnistLoader(model_path=filepath)
    mlp = model.load()

    return mlp
