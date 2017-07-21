import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = .01
batch_size = 128
n_epochs = 25
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name='bias')
logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph/03/logistic_reg', sess.graph)
    start_time = time.time()
    sess.run(init)
    n_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
        print 'Epoch %d' % i
    print 'total time: %d seconds' % (time.time() - start_time)

    # test
    writer.close()
    n_batches = int(MNIST.test.num_examples/batch_size)
    total_corr = 0.
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})
        probs = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(probs, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_corr += sess.run(accuracy)
    print 'Accuracy %d' % (total_corr/MNIST.test.num_examples)

