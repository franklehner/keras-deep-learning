#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable=all
"""
real_example_with_dropout.py
============================
"""
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils


np.random.seed(1671)


NB_EPOCH = 250
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

(X_train, y_train), (X_test, y_test) = mnist.load_data()

RESHAPE = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(60000,RESHAPE)
X_test = X_test.reshape(10000, RESHAPE)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPE,)))
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=OPTIMIZER,
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCH,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
