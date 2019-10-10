"""
Test for the imdb entity
"""


import numpy as _np
import pytest as _pytest

import lib.domain.entities.imdb as _imdb


@_pytest.mark.parametrize("label", [0, 1])
def test_constructor(label):
    """
    Test if the constructor works
    """
    sequence = [1, 2, 3]
    imdb = _imdb.Critic(sequence, label)

    assert isinstance(imdb.sequence, list)
    assert imdb.sequence == sequence
    assert imdb.label == label


def test_constructor_with_wrong_sequence_format():
    """
    Test if the constructor raises an Error
    """
    sequence = "Hallo"
    label = 1
    with _pytest.raises(TypeError):
        _ = _imdb.Critic(sequence, label)


@_pytest.mark.parametrize("label", ["hallo", 1.5])
def test_constructor_with_wrong_label_format(label):
    """
    Test if the constructor raises an Error
    """
    sequence = [1, 2, 3]
    with _pytest.raises(TypeError):
        _ = _imdb.Critic(sequence, label)


def test_constructor_with_wrong_label_value():
    """
    Test whether the constructor raises an Error
    """
    sequence = [1, 2, 3]
    label = 5
    with _pytest.raises(ValueError):
        _ = _imdb.Critic(sequence, label)


def test_vectorize_sequences_default():
    """
    Test the vectorizing of the sequences
    """
    sequences = _np.random.randint(9999, size=(25000, 200))
    x_train = _imdb.Critic.vectorize_sequences(sequences)

    assert x_train.shape[0] == 25000
    assert x_train.shape[1] == 10000


def test_vectorize_sequences():
    """
    Test concrete values
    """
    sequences = _np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7],
            [8, 9],
        ],
    )

    x_train = _imdb.Critic.vectorize_sequences(sequences, dimension=20)

    assert sequences.shape[0] == 3
    assert x_train.shape[0] == 3


def test_reverse_word_index():
    """
    Test the reversed word index
    """
    word_index = {
        "foo": 1,
        "bar": 2,
    }

    reversed_word_index = _imdb.Critic.get_reverse_word_index(word_index)

    assert isinstance(reversed_word_index, dict)
    assert reversed_word_index[1] == "foo"
    assert reversed_word_index[2] == "bar"


def test_reverse_word_index_false_type():
    """
    Test reverse_word_index with type list
    """
    word_index = [("foo", 1), ("bar", 2)]

    with _pytest.raises(AttributeError):
        _ = _imdb.Critic.get_reverse_word_index(word_index)
