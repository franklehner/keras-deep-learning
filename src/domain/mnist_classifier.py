"""Classifier for mnist data"""
from dataclasses import dataclass
from typing import List, Tuple, Union

from numpy import ndarray

from src.app.dataset_reader import MnistDataSet
from src.app.net_builder import NetBuilder
from src.app.net_inputs import prepare_dataset
from src.app.yaml_reader import YamlNetwork
from src.domain.models.datasets import DataSet
from src.domain.models.net_configurations import Network
from src.domain.models.neural_network import (
    Model,
    NNFunctional,
    NNSequential,
    plot_model,
)


@dataclass
class MnistNet:
    """Network for mnist data"""

    path: str
    model_path: str

    def parse_yaml_file(self, splitted: bool = False) -> Union[Network, List[Network]]:
        """parse yaml file"""
        yaml_network = YamlNetwork()

        return yaml_network.read_network_from_yaml(
            network_path=self.path,
            splitted=splitted,
        )

    def load_mnist_data(self) -> DataSet:
        """load mnist dataset"""
        data_set = MnistDataSet()

        return data_set.load_dataset()

    def configure_dataset(
        self,
        x_train: ndarray,
        x_test: ndarray,
        net_arch: str,
    ) -> Tuple[ndarray, ndarray]:
        """Configure the dataset for the given net architecture"""
        if net_arch == "MLP":
            x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[1])
            x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[1])
        if net_arch == "CNN":
            if x_train.ndim < 4:
                x_train = x_train.reshape((-1, x_train.shape[1], x_train.shape[1], 1))
                x_test = x_test.reshape((-1, x_test.shape[1], x_test.shape[1], 1))

        return (x_train, x_test)

    def fit(
        self,
        x_train: ndarray,
        y_train: ndarray,
        validation_data: List[ndarray],
        model: Union[NNFunctional, NNSequential],
        network: Network,
    ) -> None:  # pylint: disable=too-many-arguments
        """Fit the model"""
        if isinstance(model, NNSequential):
            model.fit(
                x_train=x_train,
                y_train=y_train,
                epochs=network.epochs,
                batch_size=network.batch_size,
            )
        else:
            model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                epochs=network.epochs,
                batch_size=network.batch_size,
            )

    def evaluate(
        self,
        model: Union[NNFunctional, NNSequential],
        x_test: ndarray,
        y_test: ndarray,
        network: Network,
    ) -> None:
        """evaluate the test"""
        if isinstance(model, NNSequential):
            acc = model.evaluate(
                x_test=x_test,
                y_test=y_test,
                batch_size=network.batch_size,
            )
        else:
            _, acc = model.evaluate(
                x=x_test,
                y=y_test,
                batch_size=network.batch_size,
            )

        print(f"Test accuracy: {round(acc * 100, 2)}%")

    def save(self, model: Union[NNFunctional, NNSequential]) -> None:
        """save model"""
        model.save(filepath=self.model_path)

    def run(self, optimizer: str, loss: str, metrics: List[str]):
        """Usecase for training a model"""
        network = self.parse_yaml_file()
        assert isinstance(network, Network)
        mnist_dataset = self.load_mnist_data()
        net_builder = NetBuilder(
            model_path=self.model_path,
            network=network,
        )
        model = net_builder.parse_network()
        model.summary()
        plot_model(
            model=model,
            to_file=self.model_path.replace(".keras", ".png"),
        )
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )
        x_train, y_train, x_test, y_test = prepare_dataset(dataset=mnist_dataset)
        x_train, x_test = self.configure_dataset(
            x_train=x_train,
            x_test=x_test,
            net_arch=network.net,
        )
        self.fit(
            x_train=x_train,
            y_train=y_train,
            validation_data=[x_test, y_test],
            model=model,
            network=network,
        )

        self.evaluate(model=model, x_test=x_test, y_test=y_test, network=network)
        self.save(model=model)

    def run_with_branches(self, loss: str, optimizer: str, metrics: List[str]) -> None:
        """Run splitted networks"""
        networks = self.parse_yaml_file(splitted=True)
        assert isinstance(networks, list)
        mnist_dataset = self.load_mnist_data()
        branches = []
        for network in networks[:-1]:
            net_builder = NetBuilder(
                model_path=self.model_path,
                network=network,
            )
            branches.append(net_builder.parse_sequence_functional(network.sequence))
        network = networks[-1]
        net_builder = NetBuilder(
            model_path=self.model_path,
            network=network,
        )
        inputs = [branch[0] for branch in branches]
        outputs = [branch[1] for branch in branches]
        net_output = net_builder.concatenate_nets(
            inputs=outputs,
            sequence=network.sequence,
        )

        model = Model(inputs=inputs, outputs=net_output)
        model.summary()
        plot_model(
            model=model,
            to_file=self.model_path.replace(".keras", ".png"),
        )
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )
        x_train, y_train, x_test, y_test = prepare_dataset(dataset=mnist_dataset)
        x_train, x_test = self.configure_dataset(
            x_train=x_train,
            x_test=x_test,
            net_arch=network.net,
        )
        model.fit(
            x=[x_train for _ in range(len(branches))],
            y=y_train,
            validation_data=([x_test for _ in range(len(branches))], y_test),
            epochs=network.epochs,
            batch_size=network.batch_size,
        )

        score = model.evaluate(
            [x_test for _ in range(len(branches))],
            y_test,
            batch_size=network.batch_size,
        )

        print(f"\nTest accuracy: {round(100 * score[1], 2)}%")
        self.save(model=model)
