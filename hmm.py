import tensorflow as tf
import numpy as np


def transition(state, trans_mat):
    acc = 0
    r = np.random.random()
    for i, prob in enumerate(trans_mat[state]):
        acc += prob
        if r < acc:
            return i


def emit(state, em_mat):
    acc = 0
    r = np.random.random()
    for i, prob in enumerate(em_mat[state]):
        acc += prob
        if r < acc:
            return i

# DEFINE ACTUAL MATRICES, CREATE DATA
trans_mat = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.3, 0.6],
    [0.3, 0.6, 0.1]
])
states_data = np.zeros(10, dtype=np.int32)
states_data[0] = 0
for i in range(1, 10):
    new_state = transition(states_data[i-1], trans_mat)
    states_data[i] = new_state
em_mat = np.array([
    [0.8, 0.2],
    [0.2, 0.8],
    [0.4, 0.6]
])
obs_data = np.array(map(lambda state: emit(state, em_mat), states_data))
obs_data_mat = np.zeros([len(obs_data), 2])
for i, o in enumerate(obs_data):
    obs_data_mat[i, o] = 1

# EXTRACT PARAMS
T = len(obs_data)
N_OBS = 2
N_H = 3

# CREATE COMPUTATION GRAPH
A = tf.Variable(tf.truncated_normal([N_H, N_H]), name='trans_mat')
v = tf.Variable(tf.truncated_normal([N_H, N_OBS]), name='em_mat')

with tf.name_scope("states"):
    states = [tf.Variable(tf.nn.softmax(tf.truncated_normal([1, 3])))]
    for i in range(T):
        next_state = tf.matmul(states[i-1], A, name="state%d" % i)
        states.append(next_state)
with tf.name_scope("preds"):
    preds = []
    for j in range(T):
        pred = tf.matmul(states[j], v, name="obs%d" % j)
        preds.append(pred)

obs = tf.placeholder(tf.float32, shape=[T, N_OBS])
with tf.name_scope("entropies"):
    entropies = []
    for i in range(T):
        entropy = obs[i] * tf.log(preds[i])
        entropies.append(entropy)
entropy = -tf.add_n(entropies)
optimizer = tf.train.AdamOptimizer().minimize(entropy)


N_EPOCHS = 3
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i, epoch in enumerate(range(N_EPOCHS)):
        A_val, v_val, pred_val = sess.run([A, v, preds])
        print "A=", A_val
        print "v=", v_val
        print "preds=", pred_val
        _, loss = sess.run([optimizer, entropy], feed_dict={obs: obs_data_mat})
        print "loss=", loss
        # print "Loss at epoch %d = %.3f" % (i, loss)
    writer = tf.summary.FileWriter('./hmm-graphs/graph1', sess.graph)
    writer.close()
