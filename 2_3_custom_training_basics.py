# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

x = tf.zeros([10,10])

x+=2

print(x)

v = tf.Variable(1.0)
print(v)

# Re-assign the value
v.assign(3.0)
print(v)

# Use 'v' in a tensorflow operation like tf.square() and reassign
v.assign( tf.square(v) )
print(v)

# Example : Fitting a linear model
# f(x) = x*W + b

class Model(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values
        
        self.W = tf.Variable(19.0)
        self.b = tf.Variable(89.0)
        
    def __call__(self, x):
        return self.W*x +self.b

model = Model()

print( model(3) )

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W +TRUE_b + noise

import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c = 'b')
plt.scatter(inputs, model(inputs), c = 'r')
plt.show()

print('Current loss : ')
print(loss(model(inputs), outputs).numpy())

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
        
        dW, db = t.gradient(current_loss, [model.W, model.b])
        model.W.assign_sub(learning_rate*dW)
        model.b.assign_sub(learning_rate*db)
 
# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(50)

for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    train(model, inputs, outputs, 0.1)
    current_loss = loss(model(inputs), outputs)
    print("Epoch %2d: W=%1.2f, b=%1.2f, loss=%2.5f'" %(epoch, Ws[-1], bs[-1], current_loss ) )

# Let's plot it all
    
plt.plot(epochs, Ws, 'r', 
         epochs, bs, 'b')

plt.plot([TRUE_W]*len(epochs), 'r--', 
         [TRUE_b]*len(epochs), 'b--')

plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()


































