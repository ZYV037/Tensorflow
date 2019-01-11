# -*- coding: utf-8 -*-
import tensorflow as tf

tf.enable_eager_execution()

# Gradient tapes
x = tf.ones((2,2))
print(x)

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    print(y)
    z = tf.multiply(y,y)
    print(z)
    
# Derivative of z with respect to the original input tensor x
print("z = ", z)
print("x = ", x)
dz_dx = t.gradient(z, x)
print("dz_dx = ", dz_dx)
for i in  [0,1]:
    for j in [0,1]:
        print("i = ", i, "j = ", j , "dz_dx[i][j] = ",dz_dx[i][j].numpy())
        assert dz_dx[i][j].numpy() ==8.0

x = tf.constant(3.0)        
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y=x*x
    z=y*y
    
dz_dx = t.gradient(z, x)
print(dz_dx)
dy_dx = t.gradient(y, x)
print(dy_dx)
del t # Drop the reference to the tap

def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
        
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

print( grad(x, 100).numpy() )
print( grad(x, 6).numpy() )
print( grad(x, 5).numpy() )
print( grad(x, 4).numpy() )

x = tf.Variable(1.0)
with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x*x*x
        # Compute the gradient inside the 't' context manager
        # which means the gradient computation is diffenentiable as well
        
        dy_dx = t2.gradient(y, x)
        d2y_dx2 = t.gradient(dy_dx, x)
        print( dy_dx.numpy() )
        print( d2y_dx2.numpy() )































