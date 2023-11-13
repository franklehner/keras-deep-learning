"""Configuration for the network
"""
from dataclasses import dataclass
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


@dataclass
class Network:
    """Network parameters
    """
    sequence: List[Layer]
    batch_size: int
    epochs: int
    net: str
