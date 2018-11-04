#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=all



import matplotlib as mpl
mpl.use("Agg")
import pandas as pd
import numpy as np
import os

from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Activation
from keras.layers import BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.regularizers import l1, l2
from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras_adversarial import fix_names
import keras.backend as K
from keras.datasets import cifar10
