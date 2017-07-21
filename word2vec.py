import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

BATCH_SIZE = 128
VOCAB_SIZE = 50000
EMBED_SIZE = 128
SKIP_WINDOW = 1
NUM_SAMPLED = 64
LEARNING_RATE = 1.0
LOGDIR = 'graph'
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000


def word2vec(batch_gen):
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='target_words')

    with tf.name_scope('embed'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
                                   name='embed_matrix')

    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                     stddev=1.0/EMBED_SIZE**.5),
                                 name='nce_weight')
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
        # global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        sess.run(tf.global_variables_initializer())
        average_loss = 0.0
        for i in xrange(NUM_TRAIN_STEPS):
            batch = batch_gen.next()
            loss_batch, _ = sess.run([loss, optimizer],
                                     feed_dict={center_words: batch[0], target_words: batch[1]})
            average_loss += loss_batch
            if (i + 1) % 2000 == 0:
                print 'Average loss at step %d: %5.1f' % (i+1, average_loss/(i+1))
                average_loss = 0.

        final_embed_matrix = sess.run(embed_matrix)
        embedding_var = tf.Variable(final_embed_matrix[:500], name="embedding")
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        writer = tf.summary.FileWriter(LOGDIR)
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = LOGDIR + "/vocab.tsv"
        projector.visualize_embeddings(writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, LOGDIR + "/skip-gram.ckpt", 1)
        writer.close()


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

