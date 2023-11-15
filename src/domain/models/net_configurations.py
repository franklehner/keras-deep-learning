"""Configuration for the network
"""
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

DenseShape = int
InputShape = Tuple[int, int, int]


@dataclass
class Layer:  # pylint: disable=too-many-instance-attributes
    """Layer"""
    name: str
    units: Optional[int] = None
    dropout: Optional[float] = None
    input_shape: Optional[InputShape] = None
    input_dim: Optional[DenseShape] = None
    activation: Optional[str] = None
    rate: Optional[float] = None
    kernel_size: Optional[int] = None
    filters: Optional[int] = None
    pool_size: Optional[Tuple[int, int]] = None
    flatten: Optional[str] = None
    dilation_rate: Optional[float] = None
    padding: Optional[str] = None

    def to_dict(self):
        """convert layer to dict for given values"""
        params = {
            k: v for k, v in asdict(self).items()
            if v is not None
        }
        params.pop("name")

        return params


@dataclass
class Network:
    """Network parameters
    """
    sequence: List[Layer]
    batch_size: int
    epochs: int
    net: str
    network_type: str
