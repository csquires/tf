import tensorflow as tf
import os


def huber_loss(labels, predctions, delta=1.0):
    residual = tf.abs(predctions - labels)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)


def make_dir(path):
    try:
        os.mkdir(path)
    except OSError: pass