"""
Test infrastructure for imdb database
"""


import pytest as _pytest

import lib.infrastructure.databases.imdb as _infra


def get_maximum_index(train_data):
    """
    Get the maximum word index from train_data
    """
    return max([max(sequence) for sequence in train_data])


@_pytest.mark.parametrize("num_words", [10000, 10])
def test_load(num_words):
    """
    Test the load function
    """
    (train_data, train_label), (test_data, test_label) = _infra.CriticRepository().load(num_words)

    assert get_maximum_index(train_data) == num_words - 1
    assert train_label.max() == 1
    assert get_maximum_index(test_data) == num_words - 1
    assert test_label.max() == 1


def test_load_all():
    """
    Test the load all function
    """
    (train_data, train_label), (test_data, test_label) = _infra.CriticRepository().load_all()

    assert get_maximum_index(train_data) == 88586
    assert train_label.max() == 1
    assert get_maximum_index(test_data) == 88584
    assert test_label.max() == 1


def test_get_word_index():
    """
    Get the word index of the words
    """
    word_index = _infra.CriticRepository().get_word_index()

    assert isinstance(word_index, dict)
