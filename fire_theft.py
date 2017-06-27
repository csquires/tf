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

Y_predicted = X*w + b
loss = tf.square(Y - Y_predicted, name='loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for x, y in data:
            sess.run(optimizer, feed_dict={X: x, Y: y})
    w_value, b_value = sess.run([w, b])
