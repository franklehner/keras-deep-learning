"""Prepare net inputs"""
from typing import Tuple

from keras.utils import to_categorical
from numpy import ndarray

from src.domain.models.datasets import DataSet


def prepare_dataset(
    dataset: DataSet, subtract_pixel_mean: bool = False,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Prepare dataset for networks"""
    y_train = to_categorical(dataset.y_train)
    y_test = to_categorical(dataset.y_test)
    x_train = dataset.x_train.astype("float32") / 255
    x_test = dataset.x_test.astype("float32") / 255

    if subtract_pixel_mean:
        x_train_mean = x_train.mean(axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    return x_train, y_train, x_test, y_test
