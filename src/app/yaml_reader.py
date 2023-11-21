"""Read from yaml files
"""
from dataclasses import dataclass
from typing import Dict, List, Union

from src.domain.models.net_configurations import Network
from src.infra.model_input import YAMLReader, parse_sequence


NetParams = Dict
Sequence = List[Dict]


@dataclass
class YamlNetwork:
    """Reader"""

    def read_network_from_yaml(
        self, network_path: str, splitted: bool = False,
    ) -> Union[Network, List[Network]]:
        """Read network from yaml file"""
        reader = YAMLReader(path=network_path)
        if splitted:
            return reader.read_splitted_network()

        return reader.read()


@dataclass
class NetworkReader:
    """Reader"""

    def read_network(self, params: NetParams) -> Network:
        """read network data"""
        sequence = params.get("Sequence")
        assert isinstance(sequence, list)
        layers = parse_sequence(sequence=sequence)

        return Network(
            sequence=layers,
            batch_size=params["batch_size"],
            epochs=params.get("epochs", 10),
            net=params["net"],
            network_type=params["network_type"],
        )
