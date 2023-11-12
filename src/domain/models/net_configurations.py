"""Configuration for the network
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

DenseShape = int
RNNShape = Tuple[int, int]
CNNShape = Tuple[int, int, int]


@dataclass
class Layer:
    """Layer"""
    name: str
    units: Optional[int] = None
    dropout: Optional[float] = None
    input_shape: Optional[RNNShape] = None
    input_dim: Optional[DenseShape] = None
    activation: Optional[str] = None


@dataclass
class Network:
    """Network parameters
    """
    sequence: List[Layer]
    batch_size: int
    epochs: int
