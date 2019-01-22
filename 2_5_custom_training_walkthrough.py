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

#tf.enable_eager_execution()

print("Tensorflow Version : {}".format( tf.__version__ ))
print("Eager execution:{}".format( tf.executing_eagerly() ))


train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]

print("column_names =", column_names)
print("feature_names =", feature_names)
print("label_name =", label_name)

class_names = ['Iris setosa', 'Isis versicolor', 'Iris vriginica']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names = column_names,
    label_name = label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))

#print(features)
#print(labels)

"""
plt.scatter(features['petal_length'].numpy,
            features['sepal_length'].numpy,
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
"""
def pack_features_vector(features, labels):
    # Pack the features into a single array.
    features = tf.stack(list(features.values()), axis=1)
    return features, labels
print('--------------------------------------------')
print(train_dataset)
print('--------------------------------------------')
train_dataset = train_dataset.map(pack_features_vector)
print(train_dataset)
print('--------------------------------------------')

features, labels = next(iter(train_dataset))

print(features[:5])
print(labels[:5])

"""
A model is a relationship between features and the labe. For the iris 
"""

model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation = tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation = tf.nn.relu),
        tf.keras.layers.Dense(3)
        ])

predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

print("Prediction:{}\n".format(tf.argmax(predictions, axis=1)))
print("Labels:{}\n".format(labels))

# Define the loss and gradient function
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

l = loss(model, features, labels)
print("Lost test: {}\n".format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Create an optimizer
# An optimizer applies the computed gradients to the model's variables to minimize the loss function.
# stochastic gradient descent(SGD) algorithm
# tf.train.GradientDescentOptimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(global_step.numpy(), loss_value.numpy() ))

optimizer.apply_gradients(zip(grads, model.variables), global_step)
print("Step: {},  Loss: {}".format(global_step.numpy(), loss(model, features, labels).numpy()))

# Training loop
## Note : rerunning this cell uses the same model variables

# keep results for plooting 
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = 0.0
    epoch_accuracy = 0.0
    #epoch_accuracy = tf.metrics.accuracy
    
    num_of_data = 0
    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables), global_step)
        
        # Track progress
        epoch_loss_avg = epoch_loss_avg + loss_value.numpy()
        #epoch_accuracy = epoch_accuracy + (tf.argmax(model(x), axis = 1, output_type=tf.int32).numpy() - y).numpy()
        #epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
        
    # end epoch
    train_loss_results.append(epoch_loss_avg)
    #epoch_accuracy.append(epoch_accuracy.result())

    #if epoch % 50 == 0:
        #print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
    
fig, axes = plt.subplots(2, sharex=True, figsize==(12,8))
fig.suptitle("Traing Metrics")

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(epoch_accuracy_results)

test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)

test_dataset = tf.data.experimental.make_csv_dataset(
        test_fp, 
        batch_size,
        cloumn_names=cloumn_names,
        label_name = 'species',
        num_epochs=1,
        shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tf.metrics.accuracy

for(x, y) in test_dataset:
    logits = model(x)
    perdition = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(predition,y)
    
print("Test set accuracy:{.3%}".format(test_accuracy.result()))

tf.stack([y, prediction], axis=1)

predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
        ])

predictions = model(predict_dataset)

for i, logits in enumeratee(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))






















































