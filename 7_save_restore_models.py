# -*- coding: utf-8 -*-

"""
https://tensorflow.google.cn/tutorials/keras/save_and_restore_models

Caution:Be careful with untrused code
TensorFlow models are code

 !pip install -q h5py pyyaml

"""

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images[:1000].reshape(-1, 28*28)/255.0
train_labels = train_labels[:1000]

test_images = test_images[:1000].reshape(-1, 28*28)/255.0
test_labels = test_labels[:1000]

def create_model():
    model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(28*28, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    return model
    


checkpoint_path = "training_1/cp.ckpt"
checkpint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only = True,
                                                 verbose=1)


checkpoint_path2 = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path2)

cp_callback2 = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path2, verbose=1, save_weights_only=True,
        period=5)

model = create_model()
model.summary()


model.fit(train_images, train_labels,
          epochs = 5,
          validation_data = (test_images, test_labels),
          #callbacks = [cp_callback2],
          verbose=0)
model.save_weights("./checkpoints/my_checkpoint")
model.save("my_model.h5")

""" 1. weihts
another_model = create_model()
loss, acc = another_model.evaluate(test_images, test_labels)
print("Untrained model, accuracy = : {:5.2f}%".format(100*acc))

another_model.load_weights("./checkpoints/my_checkpoint")
loss, acc = another_model.evaluate(test_images, test_labels)
print("After restore checkpoint, accuracy = : {:5.2f}%".format(100*acc))
"""

""" 2. h5
another_model = keras.models.load_model("my_model.h5")
loss, acc = another_model.evaluate(test_images, test_labels)
print("After restore checkpoint, accuracy = : {:5.2f}%".format(100*acc))

"""

"""3. saved model
"""
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
print(saved_model_path)
another_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
another_model.compile(optimizer="adam",
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
loss, acc = another_model.evaluate(test_images, test_labels)
print("After restore checkpoint, accuracy = : {:5.2f}%".format(100*acc))

























