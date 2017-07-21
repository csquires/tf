import tensorflow as tf
import time
import os
import tensorflow.contrib.layers as layers
import utils
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

N_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1

with tf.name_scope('data'):ze=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv2') as scope:
    X = tf.placeholder(tf.float32, [None, 784], name="X")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y")
dropout = tf.placeholder(tf.float32, name='dropout')

# conv -> relu -> pool -> conv -> relu -> pool -> fc -> softmax
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# variable scope: easy way to do namespacing (don't worry about making distinct names)
# 'weights' in conv1 becomes 'conv1-weights'
# get_variable uses variable in name space if it exists, otherwise makes new one
with tf.variable_scope('conv1') as scope:
    images = tf.reshape(X, shape=[-1, 28, 28, 1])  # -1: make total size constant
    kernel = tf.get_variable('kernel', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [32], initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv + biases, name=scope.name)
    # conv1 = layers.conv2d(images, 32, 5, 1, activation_fn=tf.nn.relu, padding='SAME')

with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(conv1, ksi
    kernel = tf.get_variable('kernels', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [64], initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases, name=scope.name)

with tf.variable_scope('pool2') as scope:
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('fc') as scope:
    input_features = 7 * 7 * 64
    w = tf.get_variable('weights', [input_features, 1024], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1024], initializer=tf.constant_initializer(0.0))
    pool2 = tf.reshape(pool2, [-1, input_features])
    fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

with tf.variable_scope('softmax_linear') as scope:
    w = tf.get_variable('weights', [1024, N_CLASSES], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES], initializer=tf.random_normal_initializer())
    logits = tf.matmul(fc, w) + b

with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy, name='loss')

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('histogram_loss', loss)
    summary_op = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
utils.make_dir('checkpoints')
utils.make_dir('checkpoints/convnet_mnist')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./graphs/convnet', sess.graph)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    initital_step = global_step.eval()
    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.
    for ind in range(initital_step, n_batches * N_EPOCHS):
        X_b, Y_b = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch, summary = sess.run([optimizer, loss, summary_op],
                                          feed_dict={X: X_b, Y: Y_b, dropout: DROPOUT})
        writer.add_summary(summary, global_step=ind)
        total_loss += loss_batch
        if (ind + 1) % SKIP_STEP == 0:
            print 'Average loss at step %d: %5.1f' % (ind + 1, total_loss / SKIP_STEP)
            total_loss = 0.
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', ind)

    print 'Optimization finished'
    print 'Total time: %.1f seconds' % (time.time() - start_time)

    n_batches = int(mnist.test.num_examples/BATCH_SIZE)
    total_corr = 0
    for i in range(n_batches):
        X_b, Y_b = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                               feed_dict={X: X_b, Y: Y_b, dropout: 1.0})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_b, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_corr += sess.run(accuracy)
    print 'Accuracy: %.2f' % (total_corr/mnist.test.num_examples)
