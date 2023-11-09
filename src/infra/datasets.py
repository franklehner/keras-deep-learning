"""Get datasets from keras
"""
from dataclasses import dataclass, field

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical


@dataclass
class Mnist:
    """
    class for the mnist data of keras
    """
    x_train: np.ndarray = field(init=False)
    y_train: np.ndarray = field(init=False)
    x_test: np.ndarray = field(init=False)
    y_test: np.ndarray = field(init=False)
    image_size: int = field(init=False)
    input_size: int = field(init=False)
    num_labels: int = field(init=False)

    def __post_init__(self):
        self._load_data()
        self.num_labels = len(np.unique(self.y_train))
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        self.image_size = self.x_train.shape[1]
        self.input_size = self.image_size * self.image_size
        self.x_train = np.reshape(self.x_train, [-1, self.input_size ])
        self.x_train = self.x_train.astype("float32") / 255
        self.x_test = np.reshape(self.x_test, [-1, self.input_size ])
        self.x_test = self.x_test.astype("float32") / 255


    def _load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
