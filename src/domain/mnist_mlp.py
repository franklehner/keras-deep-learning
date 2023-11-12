"""Usecase for mnist classification with mlp
"""
from dataclasses import dataclass

from src.app.dataset_reader import MnistDataSet
from src.app.mnist_classifier_mlp import NetBuilderMLP
from src.app.yaml_reader import YamlNetwork


@dataclass
class MLP:
    """fully dense net"""

    path: str
    model_path: str
    yaml_network: YamlNetwork
    mnist_dataset: MnistDataSet

    def run(self):
        """usecase"""
        network = self.yaml_network.read_network_from_yaml(
            network_path=self.path,
        )
        mnist_dataset = self.mnist_dataset.load_dataset()
        net_builder = NetBuilderMLP(
            model_path=self.model_path,
            dataset=mnist_dataset,
            network=network,
        )
        x_train, y_train, x_test, y_test = net_builder.prepare_dataset()
        model = net_builder.parse_network()
        model.summary()
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
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

        print(f"\n\nAccuracy: {acc * 100}%")
