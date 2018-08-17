#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
"""
build_simple_network.py

Use the MNIST dataset which comprise 70,000 examples of handwritten digits by
many different people.
"""


import argparse as _argparse
import numpy as _np

from keras.datasets import mnist as _mnist

from lib.simple_neural_network import Model as _Model


class Script:
    """
    Main Script class
    """
    def __init__(self, options):
        """
        Constructor

        :param options: parsed options from the command line
        """
        self.hidden_neurons = options.layers
        self.activations = options.activations
        self.epochs = options.epochs
        self.batch_size = options.batch_size
        self.save_file = options.save

    def run(self):
        """
        Runner method for the client
        """
        (source_train, target_train), (source_test, target_test) = _mnist.load_data()

        source_train = source_train.reshape(
            source_train.shape[0],
            source_train.shape[1] * source_train.shape[2]
        )
        source_test = source_test.reshape(
            source_test.shape[0],
            source_test.shape[1] * source_test.shape[2]
        )

        classes = _np.unique(target_train).size

        if classes != self.hidden_neurons[-1]:
            print(
                "WARNING: The last layer is not equal to the classes {}".format(
                    self.hidden_neurons[-1]
                )
            )
            self.hidden_neurons[-1] = classes
            print("Corrected automatically\n")

        model = _Model(
            input_size=source_train.shape[1],
            classes=classes,
            hidden_neurons=self.hidden_neurons,
            activations=self.activations
        )

        target_train = model.categorize_target(target_train)
        target_test = model.categorize_target(target_test)

        model.compile_loss_function()

        model.fit(
            training_source=source_train,
            training_target=target_train,
            batch_size=self.batch_size,
            epochs=self.epochs
        )

        score = model.evaluate(source_test, target_test)[1]
        print("Score: {0:.4f}".format(score))

        if self.save_file:
            self.save(model, "data/model.h5")

    @classmethod
    def save(cls, model, filepath):
        """
        Save model

        :param model: model object
        :param filepath: String Path and filename
        """
        model.save(filepath)


PARSER = _argparse.ArgumentParser(description="mnist classifier")

PARSER.add_argument(
    "-l",
    "--layers",
    type=int,
    nargs="+",
    help="List of neurons in the hidden layers",
    required=True
)

PARSER.add_argument(
    "-a",
    "--activations",
    type=str,
    nargs="+",
    help="List of activation functions",
    required=True
)

PARSER.add_argument(
    "-e",
    "--epochs",
    type=int,
    help="Number of iterations for the network",
    default=10
)

PARSER.add_argument(
    "-b",
    "--batch_size",
    type=int,
    help="Batch size",
    default=128
)

PARSER.add_argument(
    "-s",
    "--save",
    action="store_true",
    help="Save model file"
)

OPTIONS = PARSER.parse_args()


Script(OPTIONS).run()
