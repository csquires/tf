import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets('/data/mnist', one_hot=True)

learning_rate = .01
batch_size = 128
n_epochs = 25
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name='bias')
logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})

