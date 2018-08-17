#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
use_mnist_model.py
==================
"""


import argparse
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import load_model


class Script:
    """
    Main class
    """
    def __init__(self):
        """
        Constructor
        """
        self.options = self.get_options()

    def run(self):
        """
        Runner
        """
        (source_train, _), (source_test, _) = mnist.load_data()

        source_train = source_train.reshape(
            source_train.shape[0], source_train.shape[1] * source_train.shape[2]
        )
        source_test = source_test.reshape(
            source_test.shape[0], source_test.shape[1] * source_test.shape[2]
        )

        self.plot_number(source_test[self.options.idx])

        model = load_model(self.options.filename)
        predictions = model.predict(source_test[self.options.idx].reshape(1, 784))
        probability = predictions.max()
        prediction = predictions.argmax()

        print(
            "Mit einer Wahrscheinlichkeit von {0:.3f} ist es die Zahl {1}".format(
                probability,
                prediction
            )
        )

    @staticmethod
    def plot_number(vector):
        """
        Plot number
        """
        vector = vector.reshape(28, 28)
        plt.imshow(vector)
        plt.show()

    @classmethod
    def get_options(cls):
        """
        Get options
        """
        parser = argparse.ArgumentParser(description="use mnist model")

        parser.add_argument(
            "-f",
            "--filename",
            type=str,
            default="data/model.h5",
            help="Specify model file"
        )

        parser.add_argument(
            "-i",
            "--idx",
            type=int,
            default=0,
            help="Specify picture index"
        )

        return parser.parse_args()


if __name__ == "__main__":
    Script().run()
