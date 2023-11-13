"""Load model"""
import os
from dataclasses import dataclass, field
from typing import Literal, Union

from src.domain.models.neural_network import NNFunctional, NNSequential


@dataclass
class Loader:
    """Load models"""

    path: str
    model_type: Literal["sequential", "functional"]
    model: Union[NNSequential, NNFunctional] = field(init=False)

    def _verify_path(self) -> bool:
        """verify if the model to load is available"""
        if not self.path.endswith(".keras"):
            return False
        if not os.path.exists(self.path):
            return False
        if not os.path.isfile(self.path):
            return False

        return True

    def load(self):
        """load model"""
        if not self._verify_path():
            raise RuntimeError("Please Check File")
        if self.model_type == "sequential":
            self.model = NNSequential(
                path=self.path, load=True,
            )
        elif self.model_type == "functional":
            self.model = NNFunctional(
                path=self.path, load=True,
            )
        else:
            raise RuntimeError("Use the right model type")
