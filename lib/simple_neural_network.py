"""
Simple neural network with keras
"""


import keras.models as _models
import keras.layers.core as _layers_core
import keras.utils as _utils


class Model:
    """
    Class Model

    Build layers for a neural network

    Attributes
    ----------
    input_size: Integer
        Length of the input vector
    classes: Integer
        Number of disjoint classes
    hidden_neurons: list
        list of integers with the number of hidden neurons per layer.
        The length of the list is the number of the layers.
    activations: list
        List of strings with the activation functions.
        The length of the list is the number of the layers. So the length of
        the activations must be equal to the length of the hidden neurons.

    methods:
    --------
    __evaluate_net_architecture: raises RuntimeError if the architecture is not
        valid
    __make_architecture: Builds a neural network with the number of layers
        the number of neurons per layer and the activation function per layer

    fit: Fit the model

    evaluate: evaluate the metric
    """
    def __init__(self, input_size, classes, hidden_neurons=None, activations=None):
        """
        Constructor of the model

        :param input_size: Integer
            Length of the input vector
        :param classes: Integer
            Number of disjoint classes
        :param hidden_neurons: list
            List of integers with the number of hidden neurons per layer.
            The length of the list is the number of the layers.
        :param activations: list
            List of strings with the activation functions. The length of the
            list is the number of layers in the network.
            The length of the activations and the length of the hidden neurons
            must be equal.
        """
        self.input_size = input_size
        self.classes = classes
        self.hidden_neurons = hidden_neurons if hidden_neurons else [classes]
        self.activations = activations
        self.model = _models.Sequential()

        self.__evaluate_net_architecture()
        self.__make_architecture()

    def __make_architecture(self):
        """
        Builds the model.
        Add the hidden neurons and the activation functions per layer

        Parameters:
        -----------
        self
        """
        input_size = self.input_size

        for h_neurons, activation in zip(self.hidden_neurons, self.activations):
            self.model.add(_layers_core.Dense(h_neurons, input_dim=input_size))
            self.model.add(_layers_core.Activation(activation))
            input_size = h_neurons

    def __evaluate_net_architecture(self):
        """
        Raise an error if the architecture is not valid
        """
        if not isinstance(self.activations, list) or not isinstance(self.hidden_neurons, list):
            raise RuntimeError("Activations and hidden_neurons must be type list!")

        if not len(self.activations) == len(self.hidden_neurons):
            raise RuntimeError(
                "Activations and hidden neurons must have the same length!"
            )

        if not self.classes == self.hidden_neurons[-1]:
            raise RuntimeError(
                "Classes must be equal to the last hidden neuron\n",
                "Classes: {0} last hidden neurons: {1}".format(
                    self.classes,
                    self.hidden_neurons[-1]
                )
            )

    def fit(self, training_source, training_target, batch_size, epochs):
        """
        Fit method for the model

        :param training_source: array {n_samples, n_features}
            n_samples is the count of the samples
            n_features is the length of the input vector

        :param training_target: array {n_samples, n_classes}
            n_samples is the count of the samples
            n_classes is an array with the number of distinct classes
        :param batch_size: Integer
            Size of the batches per epoch
        :param epochs: Integer
            Number of iterations for training
        :return: self
        """
        self.model.fit(
            training_source,
            training_target,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        return self

    def evaluate(self, source_test, target_test, verbose=1):
        """
        Evaluates the metrics of the test sets

        :param source_test: array {n_samples, n_features}
            n_samples is the number of samples
            n_features is the length of the input vector

        :param target_test: array {n_samples, n_classes}
            n_samples is the number of samples
            n_classes is the number of distinct classes

        :param verbose: Boolean
            If True, every step is shown

        :return: list of metrics for the evaluation samples
        """
        return self.model.evaluate(source_test, target_test, verbose=verbose)

    def compile_loss_function(
            self, loss="categorical_crossentropy", metrics=None, optimizer="sgd"
        ):
        """
        Compile the loss function

        :param loss: string
            The loss function

        :param metrics: list
            list of metrics

        :param optimizer: string
            The optimizer for the loss function

        :return: self
        """
        metrics = metrics if metrics else ["accuracy"]

        if not isinstance(metrics, list):
            raise RuntimeError(
                "metrics must be in type list not {}".format(type(metrics))
            )

        return self.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    def categorize_target(self, target):
        """
        Categorize target.
        Convert target number to an encoded array

        :param target: array-like [n_samples]

        :return: Array with shape (n_samples, n_classes)
        """
        return _utils.np_utils.to_categorical(target, self.classes)

    def predict(self, source):
        """
        Predict

        :param source: array {n_samples, n_features}

        :return: A prediction for the source
        """
        return self.model.predict(source)

    def save(self, file_path):
        """
        Save model

        :param file_path: string
            path to the saving file
        """
        self.model.save(file_path)
