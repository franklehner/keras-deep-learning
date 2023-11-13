"""Read yaml files with the configuration of the net
"""
from dataclasses import dataclass
from typing import Dict, List

import yaml

from src.domain.models.net_configurations import Layer, Network


@dataclass
class YAMLReader:
    """Read yaml files"""
    path: str

    def read(self) -> Network:
        """Read from yaml file"""
        with open(self.path, "r", encoding="utf-8") as f_obj:
            result = yaml.load(stream=f_obj, Loader=yaml.FullLoader)

        sequence = self.parse_sequence(sequence=result["Sequence"])
        network = Network(
            sequence=sequence,
            batch_size=result.get("batch_size"),
            epochs=result.get("epochs"),
            net=result.get("net")
        )

        return network

    def parse_sequence(self, sequence: Dict) -> List[Layer]:
        """parse sequence"""
        params = []
        for item in sequence:
            for key, value in item.items():
                layer = Layer(
                    name=key,
                    units=value.get("units"),
                    dropout=value.get("dropout"),
                    input_dim=value.get("input_dim"),
                    input_shape=value.get("input_shape"),
                    activation=value.get("activation"),
                    flatten=value.get("flatten"),
                    kernel_size=value.get("kernel_size"),
                    filters=value.get("filters"),
                    rate=value.get("rate"),
                    pool_size=value.get("pool_size"),
                )
                params.append(layer)

        return params
                    