"""
Movie critic
"""


import abc as _abc
import numpy as _np


class Critic:
    """
    Critic
    """
    def __init__(self, sequence, label):
        """
        Constructor for a movie object
        """
        self.sequence = sequence
        self.label = label

        self.check_sequence()
        self.check_label()

    def check_sequence(self):
        """
        Check whether the sequence is type list
        """
        if not isinstance(self.sequence, list):
            raise TypeError("sequence must be type list")

    def check_label(self):
        """
        Check whether the label is type int an 0 xor 1
        """
        if not isinstance(self.label, int):
            raise TypeError("label must be integer")

        if not (self.label == 0 or self.label == 1):
            raise ValueError("Label must be 0 or 1 not {0}".format(self.label))

    @classmethod
    def vectorize_sequences(cls, sequences, dimension=10000):
        """
        Vectorize the sequence
        """
        results = _np.zeros((len(sequences), dimension))

        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1

        return results

    @classmethod
    def get_reverse_word_index(cls, word_index):
        """
        Get the reversed word index
        """
        return {value: key for key, value in word_index.items()}

    @classmethod
    def get_decoded_review(cls, reverse_word_index, train_data, num):
        """
        Get the critic itself
        """
        return " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[num]])


class AbstractCritic:
    """
    Interface for the infrastructure
    """

    __metaclass__ = _abc.ABCMeta

    @_abc.abstractmethod
    def load(self, num_words):
        """
        Load the film data
        """
        raise NotImplementedError

    @_abc.abstractmethod
    def load_all(self):
        """
        Load all data
        """
        raise NotImplementedError

    @_abc.abstractmethod
    def get_word_index(self):
        """
        Get the word index of the imdb database
        """
        raise NotImplementedError
