import tensorflow as tf
import numpy as np


def init_viterbi_vars(N, S):
    path_states = tf.Variable(tf.zeros([N, S], dtype=tf.int64), name='states_mat')
    path_scores = tf.Variable(tf.zeros([N, S], dtype=tf.float64), name='score_mat')
    states_seq = tf.Variable(tf.zeros([N], dtype=tf.int64, name='states_seq'))
    return path_states, path_scores, states_seq


class HMM(object):
    def __init__(self, T, E, T0, epsilon=0.001, max_step=10):
        T = np.array(T)
        E = np.array(E)
        T0 = np.array(T0)
        if T0.shape[0] != T.shape[0]:
            raise ValueError("T0.shape[0] != T.shape[0]")
        if E.shape[1] != T.shape[0]:
            raise ValueError("E.shape[1] != T.shape[0]")
        with tf.name_scope('initial_params'):
            with tf.name_scope('scalar_constants'):
                self.max_step = max_step
                self.epsilon = epsilon  # convergence
                self.S = T.shape[0]  # num states
                self.O = E.shape[0]  # num observations
                self.prob_state_1 = []
            with tf.name_scope('model_params'):
                self.E = tf.Variable(E, dtype=tf.float64, name='emission_mat')
                self.T = tf.Variable(T, dtype=tf.float64, name='transition_mat')
                self.T0 = tf.Variable(tf.constant(T0, dtype=tf.float64, name='initial_state_vec'))

    def belief_propagation(self, scores):
        reshaped_scores = tf.reshape(scores, (-1, 1))
        return tf.add(reshaped_scores, tf.log(self.T))

    def viterbi_inference(self, obs_seq):
        self.N = len(obs_seq)
        obs_seq_tf = tf.constant(obs_seq, dtype=tf.int32, name='observation_sequence')

        with tf.name_scope('init_viterbi_vars'):
            path_states, path_scores, states_seq = init_viterbi_vars(self.N, self.S)

        with tf.name_scope('emission_seq'):
            obs_prob_seq = tf.log(tf.gather(self.E, obs_seq_tf))
            obs_prob_list = tf.split(obs_prob_seq, self.N, axis=0)

        with tf.name_scope('starting_log_priors'):
            # scatter_update updates path_scores at index 0 to its old value + log(T0)
            path_scores = tf.scatter_update(path_scores, 0, tf.log(self.T0) + tf.squeeze(obs_prob_list[0]))

        with tf.name_scope('belief_prop'):
            for step, obs_prob in enumerate(obs_prob_list[1:]):
                with tf.name_scope('belief_prop_step_%s' % step):
                    belief = self.belief_propagation(path_scores[step, :])
                    path_states = tf.scatter_update(path_states, step+1, tf.argmax(belief, 0))
                    path_scores = tf.scatter_update(path_scores, step+1, tf.reduce_max(belief, 0) + tf.squeeze(obs_prob))
                with tf.name_scope('max_likelihood_update'):
                    best_path = tf.arg_max(path_scores[self.N - 1, :], 0)
                    states_seq = tf.scatter_update(states_seq, self.N-1, best_path)

        with tf.name_scope('backtrack'):
            for step in range(self.N - 1, 0, -1):
                with tf.name_scope('backtrack_step_%s' % step):
                    state = states_seq[step]
                    idx = tf.reshape(tf.stack([step, state]), [1, -1])
                    state_prob = tf.gather_nd(path_states, idx)
                    states_seq = tf.scatter_update(states_seq, step-1, state_prob[0])

        return states_seq, tf.exp(path_scores)

    def run_viterbi(self, obs_seq, summary=False):
        state_graph, state_prob_graph = self.viterbi_inference(obs_seq)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print obs_seq
            print tf.gather(self.E, obs_seq).eval()
            states_seq, state_prob = sess.run([state_graph, state_prob_graph])
            if summary:
                summary = tf.summary.FileWriter('./hmm_class/', graph=sess.graph)

        return states_seq, state_prob


