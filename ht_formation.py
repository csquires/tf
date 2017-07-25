import tensorflow as tf

T = 10  # number of time steps
N = 100  # number of nodes
H = 10  # number of hashtags
D = tf.placeholder(tf.float64, shape=[T, N, H], name="D")
A = tf.placeholder(tf.float64, shape=[N, N], name="A")
alpha = tf.Variable(tf.zeros([T, 2], dtype=tf.float64), name="alpha")
logits = tf.Variable(tf.zeros([T, N, H], dtype=tf.float64), name="logits")

for t in range(1, T):
    for i in range(N):
        for h in range(H):
            weight1 = alpha[t, 0] * D[t-1, i, h]
            sims = tf.reduce_sum(D[t-1, i] * D[t-1], axis=1)  # broadcast ht vector of i onto ht vectors of all nodes
            sim_sum = tf.reduce_sum(A[i] * (D[t-1, :, h] * sims))  # broadcast sim vector onto whether each ht was used
            weight0 = alpha[t, 1] * (1 - D[t-1, i, h]) * sim_sum
            logits = tf.scatter_update(logits, [[t, i, h]], [weight1 + weight0])
