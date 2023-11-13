"""Model loader"""
import re
from typing import Optional, Union

from src.domain.models.neural_network import NNFunctional, NNSequential
from src.infra.model_loader import Loader

REGEX = re.compile(r"sequential|functional")


def load_model(path: str) -> Optional[Union[NNFunctional, NNSequential]]:
    """load a sequential model"""
    if not REGEX.search(path):
        raise RuntimeError("No valid model name")
    model_types = REGEX.findall(path)
    if len(model_types) > 1:
        raise RuntimeError("No valid model name")

    loader = Loader(
        path=path, model_type=model_types[0],
    )
    try:
        loader.load()
        model = loader.model
    except RuntimeError:
        model = None

    return model
