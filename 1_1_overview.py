import tensorflow as tf
import numpy as np 

np.random.seed(1989)
x_data = np.float32(np.random.rand(2,100))

print("---------------------")
print(x_data)
print("---------------------")

print("x_data.shape", x_data.shape)
print("x_data.ndim", x_data.ndim)
print("x_data.size", x_data.size)
print("x_data.dtype", x_data.dtype)
print("x_data.itemsize", x_data.itemsize)
print("type(x_data)", type(x_data))

y_data = np.dot([0.100, 0.200], x_data) + 0.300

print("---------------------")
print(y_data)
print("---------------------")

print(np.zeros([2]))

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(0, 201):
	sess.run(train)
	if step % 20 == 0:
		print (step, sess.run(W), sess.run(b))
		# print("x=", x_data)
		# print("y=", y_data)