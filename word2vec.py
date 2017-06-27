import tensorflow as tf

BATCH_SIZE = 128
VOCAB_SIZE = 10
EMBED_SIZE = 10
NUM_SAMPLED = 10

with tf.name_scope('data'):
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='target_words')

with tf.name_scope('embed'):
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name='embed_matrix')

with tf.name_scope('loss'):
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
    nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/EMBED_SIZE**.5), name='nce_weight')
    nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_weight,
        biases=nce_bias,
        labels=target_words,
        inputs=embed,
        num_sampled=NUM_SAMPLED,
        num_classes=VOCAB_SIZE
    ), name='loss')

optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    average_loss = 0.0
    for i in xrange(NUM_TRAIN_STEPS):
        batch = batch_gen.next()
        loss_batch, _ = sess.run([loss, optimizer], feed_dict={center_words: batch[0], target_words: batch[1]})
        average_loss += loss_batch
        if (index + 1) % 2000 == 0:
            print 'Average loss at step %d: %5.1f' % (i+1, average_loss/(i+1))

