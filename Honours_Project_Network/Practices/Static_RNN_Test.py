'''
Single Layer idetical in structure to the previous, this time utilizing the Tensorflow backend mechanics Unrolled RNN

Single layer Recurrent Neural Network
Network is comprised of five neurons in a single hidden layer,
testing purposes demonstrate a working example RNN model

[Input[1][2][3]]--->[Hidden_Layer[1][2][3][4][5]]--->[Output]

ERROR
'''

import tensorflow as tf#Import Tensorflow
import numpy as np#Import Numpy

#Variables
n_inputs = 3#Input Number Vector
n_neurons = 5#Hidden Layer Neuron Count
n_steps = 3
'''
x0 = tf.placeholder(tf.float32, [None, n_inputs])#Input 1
x1 = tf.placeholder(tf.float32, [None, n_inputs])#Input 2

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)#RNN Cell Construct
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [x0,x1], dtype = tf.float32)#Static
'''
#y0,y1 = output_seqs#Output

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
x_seqs = tf.unstack(tf.transpose(X, perm = [1,0,2]))

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs,  states = tf.contrib.rnn.static_rnn(basic_cell, x_seqs, dtype=tf.float32)

init = tf.global_variables_initializer()#TF Init args

outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])

x_batch = np.array([
    [[0,1,2], [9,8,7]],
    [[3,4,5], [0,0,0]],
    [[6,7,8], [6,5,4]],
    [[9,0,1], [3,2,1]],
    ])

with tf.Session() as sess:
    init.run()
    output_val = outputs.eval(feed_dict={X: x_batch})
    print(output_val)