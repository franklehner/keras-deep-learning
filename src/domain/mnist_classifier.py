"""Classifier for mnist data"""
from dataclasses import dataclass
from typing import List

from src.app.dataset_reader import MnistDataSet
from src.app.mnist_classifier import NetBuilder
from src.app.yaml_reader import YamlNetwork


@dataclass
class MnistNet:
    """Network for mnist data"""

    path: str
    model_path: str
    yaml_network: YamlNetwork
    mnist_dataset: MnistDataSet

    def run(self, optimizer: str, loss: str, metrics: List[str]):
        """Usecase for training a model"""
        network = self.yaml_network.read_network_from_yaml(
            network_path=self.path,
        )
        mnist_dataset = self.mnist_dataset.load_dataset()
        net_builder = NetBuilder(
            model_path=self.model_path,
            dataset=mnist_dataset,
            network=network,
        )
        x_train, y_train, x_test, y_test = net_builder.prepare_dataset()
        if network.net == "MLP":
            x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[1])
            x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[1])
        if network.net == "CNN":
            if x_train.ndim < 4:
                x_train = x_train.reshape(
                    (-1, x_train.shape[1], x_train.shape[1], 1)
                )
                x_test = x_test.reshape(
                    (-1, x_test.shape[1], x_test.shape[1], 1)
                )

        model = net_builder.parse_network()
        model.summary()
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )
        model.fit(
            x_train=x_train,
            y_train=y_train,
            epochs=network.epochs,
            batch_size=network.batch_size,
        )

        acc = model.evaluate(
            x_test=x_test,
            y_test=y_test,
            batch_size=network.batch_size,
        )
        model.save()

        print(f"Accuracy: {acc * 100}%")
