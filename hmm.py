import tensorflow as tf

T = 10
N_H = 3
N_OBS = 2

states = []
obs = []
states.append(tf.placeholder(tf.float32, shape=[1, N_H], name="state"))
A = tf.placeholder(tf.float32, shape=[N_H, N_H], name="A")
v = tf.placeholder(tf.float32, shape=[N_H, N_OBS], name="v")


for i in range(T):
    next_state = tf.matmul(states[i], A)
    states.append(next_state)
for j in range(T):
    ob = tf.matmul(states[j], v)
    obs.append(ob)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./hmm-graphs/graph1', sess.graph)
    writer.close()
