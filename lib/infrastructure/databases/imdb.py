"""
Infrastructure for imdb
"""


import keras.datasets.imdb as _imdb

import lib.domain.entities.imdb as _entity


class CriticRepository(_entity.AbstractCritic):
    """
    Critic Repo
    """

    def load(self, num_words=10000):
        """
        Load num_words of the imdb data
        """
        return _imdb.load_data(num_words=num_words)

    def load_all(self):
        """
        Load all words in the imdb data
        """
        return _imdb.load_data()

    def get_word_index(self):
        """
        Get the word index of the imdb
        """
        return _imdb.get_word_index()
