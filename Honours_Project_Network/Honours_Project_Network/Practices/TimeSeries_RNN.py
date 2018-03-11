'''
Timesteps for Time Series Prediction
'''
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os

curr_point = 0
attrs = []

t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

#df = pd.read_csv('test.csv', sep=',',header=None)

def parse_data(filename):

    f = open(filename, 'r').readlines()

    #For each row in file, do;
    for row in f:
     row = [x.strip() for x in row.split(',')]
     row = [int(num) for num in row]
     target.append(int(row[0]))
     attrs.append(row[1:])

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)




#Variables
n_steps = 20#Run Network 20 times for 20 outputs ect.
n_inputs = 1#Number of Inputs
n_neurons = 100#Neuron Count
n_outputs = 1#Number of Outputs
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

learning_rate = 0.001

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

#cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.sigmoid),
    output_size=n_outputs
    )

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#Execution Phase
n_interations = 1500
batch_size = 50
'''
with tf.Session() as sess:
    init.run()
    for interation in range(n_interations):
        x_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
        curr_point += 1
        if interation % 100 ==0:
            mse = loss.eval(feed_dict={x: x_batch, y: y_batch})
            print(interation, "\tMSE: ", mse)
'''
with tf.Session() as sess:
    init.run()
    for iteration in range(n_interations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={x: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={x: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={x: X_new})
    print("y_pred: ", y_pred)
    print("x_new:", X_new)

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

save_fig("time_series_pred_plot")
plt.show()

reset_graph()

