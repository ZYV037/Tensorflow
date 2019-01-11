# -*- coding: utf-8 -*-

#Explore overfitting and underfitting

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words = NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    #Create an all-zero matrix of shape( len(sequences), dimension )
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices]  = 1.0
    return results

#print(train_data[0])
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

#print(train_data[0])
#plt.plot(train_data[0])

def plot_history(histroies, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))    
    
    for name, history in histroies:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + " val")
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label = name.title() + " Train")
        
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    

baseline_model = keras.Sequential([
    #'imput_shape' is only required here so that '.summary' works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid) 
])
    
baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

print(baseline_model.summary())

small_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
    
small_model.compile( optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'binary_crossentropy'])

print(small_model.summary())
    


big_model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
    
big_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])

print(big_model.summary())


l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation = tf.nn.relu, input_shape = (NUM_WORDS, )),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation = tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
    
l2_model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])


dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation = tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation = tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = tf.nn.sigmoid)    
])

dpt_model.compile( optimizer = 'adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])

#*********************************model_fit************************************

baseline_history = baseline_model.fit( train_data,
                                       train_labels,
                                       epochs=20,
                                       batch_size=512,
                                       validation_data=(test_data, test_labels),
                                       verbose=2)

"""small_history = small_model.fit(train_data,
                                train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)
    
big_history = big_model.fit(train_data,
                            train_labels,
                            epochs=20,
                            batch_size=512,
                            validation_data=(test_data, test_labels),
                            verbose=2)    

l2_model_history = l2_model.fit(train_data, 
                               train_labels,
                               epochs=20,
                               batch_size=512,
                               validation_data=(test_data, test_labels),
                               verbose=2)
"""

dpt_model_history = dpt_model.fit( train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

    
plot_history([
                ('baseline', baseline_history),
             #('smaller', small_history),
             #('bigger', big_history),
             #('l2', l2_model_history),
             ('Dropout', dpt_model_history)
             ])        
    
    
    
    
    
    
    
    
    
    
    
    
    
    