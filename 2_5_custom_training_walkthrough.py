# -*- coding: utf-8 -*-
"""
categorize iris flowers by specise

1. Build a model
2. Train this model on example data
3. Use the model to predictions about unknown data

"""

# !pip install -q tf-nightly

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

tf.enable_eager_execution()

print("Tensorflow Version : {}".format( tf.__version__ ))
print("Eager execution:{}".format( tf.executing_eagerly() ))


train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))


