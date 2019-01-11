# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:41:20 2019

@author: Aaron
"""

#https://tensorflow.google.cn/tutorials/keras/basic_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#caution:
#  base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/' is invalid
# you could replace it with 'https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion/'
# or you can hander download the t10k-images-idx3-ubyte.gz... file to C:\Users\Aaron\.keras\datasets\fashion-mnist

#from tensorflow.examples.tutorials.mnist import input_data
#MNIST_data_folder = "C:/Users/Aaron/Desktop/tensorflow_learning/fashion-mnist-master"
#fashion_minist = input_data.read_data_sets( MNIST_data_folder, one_hot = True)
#train_images, train_labels = fashion_mnist.train.images, fashon_minst.train.labels
#test_images, test_labels = fashion_mnist.test.images, fashon_minst.test.labels

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print( len(train_images))
print(train_labels.shape)
print(train_labels)

def plot_image_show(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(True)

train_images = train_images/255.0
test_images = test_images/255.0

def plot_image_show_some(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[labels[i]])

#Build the Model
model = keras.Sequential([
    keras.layers.Flatten( input_shape=(28,28) ),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    
model.compile(  optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy = ", test_acc )

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(class_names[np.argmax(predictions[0])])

def plot_image(i, predictions_arrays, true_labels, imgs):
    predictions_array, true_label, img = predictions_arrays[i], true_labels[i], imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color='blue'
    else:
        color='red'
        
    plt.xlabel("{} {:2.0f}% ({}))".format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]),
                                          color=color)


def plot_value_array(i, predictions_arrays, true_labels):
    predictions_array, true_label = predictions_arrays[i], true_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3

num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)







