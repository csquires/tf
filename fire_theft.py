import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd


DATA_FILE = 'fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = .5 * tf.square(residual)
    large_res = delta * residual - .5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

Y_predicted = X*w + b
loss = huber_loss(Y, Y_predicted)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graph/03/linear_reg', sess.graph)
    for i in range(100):
        for x, y in data:
            sess.run(optimizer, feed_dict={X: x, Y: y})
        print 'Epoch %d' % i
    w_value, b_value = sess.run([w, b])

X, Y = data.T[0], data.T[1]
plt.ion()
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X*w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
