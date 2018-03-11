#Main RNN from O'Reilly

'''
Single layer Recurrent Neural Network
Network is comprised of five neurons in a single hidden layer,
testing purposes demonstrate a working example RNN model

[Input[1][2][3]]--->[Hidden_Layer[1][2][3][4][5]]--->[Output]

This design avoids utilizing the deeper functions of Tensorflow Backend infustructure to allow a more in depth understanding.
'''

import tensorflow as tf#Import Tensorflow
import numpy as np#Import Numpy

#Variables
n_inputs = 3#Input Number Vector
n_neurons = 5#Hidden Layer Neuron Count

x0 = tf.placeholder(tf.float32, [None, n_inputs])#Input 1
x1 = tf.placeholder(tf.float32, [None, n_inputs])#Input 2

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))#Weight 1
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))#Weight 2
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))#Biases

Y0 = tf.tanh(tf.matmul(x0, Wx) + b)#(Activation Function(Summation Function))
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(x1, Wx) + b)#(Activation Function(Summation Function + Previous Summation Function))

init = tf.global_variables_initializer()#TF Init args

'''
Structurally Identical to a two-layer feedforward network, RNN share the same WEIGHTS & BIASES terms are shared by both layers and 
INPUTS are feed to each layer and get OUTPUTS from each layer
'''

#Mini-Batch
x0_batch = np.array([[0,1,2],[3,4,5],[9,0,1]])#t=0
x1_batch = np.array([[9,8,7],[0,0,0],[3,2,1]])#t=1

with tf.Session() as sess:
    init.run()
    y0_val, y1_val = sess.run([Y0, Y1], feed_dict={x0: x0_batch, x1: x1_batch})
    #Network outputs Five Neuron Calcuation instances for each timestep
    print(y0_val)
    print(y1_val)