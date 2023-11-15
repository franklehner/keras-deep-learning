"""Read from yaml files
"""
from dataclasses import dataclass
from typing import List, Union

from src.domain.models.net_configurations import Network
from src.infra.model_input import YAMLReader


@dataclass
class YamlNetwork:
    """Reader"""

    def read_network_from_yaml(
        self, network_path: str, splitted: bool,
    ) -> Union[Network, List[Network]]:
        """Read network from yaml file"""
        reader = YAMLReader(path=network_path)
        if splitted:
            return reader.read_splitted_network()

        return reader.read()
