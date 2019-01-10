import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

print( len(train_data), ", ", len(train_labels))

print(train_data[0])

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items() }
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reserve_word_index = dict([(value,key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reserve_word_index.get(i, "?") for i in text])

print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), ", ", len(test_data[0]))
print(train_data[0])

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val=train_data[:10000]
pratial_x_train = train_data[10000:]

y_val=train_labels[:10000]
partial_y_train=train_labels[10000:]

history=model.fit(pratial_x_train,
                  partial_y_train,
                  epochs=40,
                  batch_size=512,
                  validation_data=(x_val,y_val),
                  verbose=1)

rest=model.evaluate(test_data, test_labels)

history_dict=history.history
print(history_dict.keys())

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
#b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validaton loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.clf()
plt.plot(epochs, acc, 'bo', label="Training acc")
plt.plot(epochs, val_acc, 'b', label="Validation acc")
plt.title("Traing and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()