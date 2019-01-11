# -*- coding: utf-8 -*-

import tensorflow as tf
import timeit

tf.enable_eager_execution()

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.encode_base64("Hello World!"))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

a = tf.constant([[1]])
print(a)
b = tf.constant([[2,3]])
print(b)
x = tf.matmul(a, b)
print(x)
print(x.shape)
print(x.dtype)

import numpy as np

nd_array = np.ones([3,3])
print(nd_array)
print("Tensorflow operation convert numpy arrays to Tensors automatically")
ts_array = tf.multiply(nd_array, 42)
print(ts_array)

print("And Numpy operations convert Tensors to numpy arrays automatically")
print(np.add(ts_array, 1))

print("The .numpy() method explicitly convert A Tensor to a numpy array")
print(ts_array.numpy())

# Gpu acceleration
x = tf.random_uniform([3,3])
print("Is there a GPU available: ")
print(tf.test.is_gpu_available())

print("Is the tensor on GPU #0: ")
print(x.device.endswith('GPU:O'))

print(tf.device)

def f(x):
    return x**2
def g(x):
    return x**4
def h(x):
    return x**8

def time_matmul(x):
    #print(timeit.timeit('[func(42) for func in (f,g,h)]', globals=globals()))
    #print( timeit.timeit( 'tf.matmul(x, x)', setup="import tensorflow as tf", globals=globals() ))
    tf.matmul(x,x)
    
#Force execution on CPU
print("On CPU: ")

with tf.device("CPU:0"):
    x = tf.random_uniform([1000,1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

#Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):#Or GPU:1 for the 2nd GPU...
        x = tf.random_uniform([1000,1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
        
ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
print(ds_tensors)

# Create a CSV file
import tempfile

_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3
""")
    
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

print("Elements of ds_tensors: ")
for x in ds_tensors:
    print(x)
    
print("\nElements in ds_file: ")
for x in ds_file:
    print(x)











































