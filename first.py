import tensorflow as tf
a = tf.constant([2, 2], name='vector')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.add(a, b, name='add')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print sess.run(x)

writer.close()
