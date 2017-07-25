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

    def initialize_fb_variables(self, shape):
        self.forward = tf.Variable(tf.zeros(shape, dtype=tf.float64), name='forward')
        self.backward = tf.Variable(tf.zeros(shape, dtype=tf.float64), name='backward')
        self.posterior = tf.Variable(tf.zeros(shape, dtype=tf.float64), name='posterior')

    def _forward(self, obs_prob_list):
        with tf.name_scope('init_scaling_factor'):
            self.scale = tf.Variable(tf.zeros([self.N], tf.float64))
        with tf.name_scope('forward_first_step'):
            init_prob = tf.multiply(self.T0, tf.squeeze(obs_prob_list[0]))
            self.scale = tf.scatter_update(self.scale, 0, 1.0/tf.reduce_sum(init_prob))
            self.forward = tf.scatter_update(self.forward, 0, self.scale[0] * init_prob)
        for step, obs_prob in enumerate(obs_prob_list[1:]):
            with tf.name_scope('time_step_%s' % step):
                prev_prob = tf.expand_dims(self.forward[step, :], 0)
                prior_prob = tf.matmul(prev_prob, self.T)
                forward_score = tf.multiply(prior_prob, tf.squeeze(obs_prob))
                forward_prob = tf.squeeze(forward_score)
                self.scale = tf.scatter_update(self.scale, step+1, 1.0/tf.reduce_sum(forward_prob))
                self.forward = tf.scatter_update(self.forward, step+1, self.scale[step+1] * forward_prob)

    def _backward(self, obs_prob_list):
        with tf.name_scope('backward_last_step'):
            self.backward = tf.scatter_update(self.backward, 0, self.scale[self.N-1] * tf.ones([self.S], dtype=tf.float64))

        for step, obs_prob in enumerate(obs_prob_list[:-1]):
            with tf.name_scope('time_step_%s' % step):
                next_prob = tf.expand_dims(self.backward[step, :], 1)
                obs_prob_d = tf.diag(tf.squeeze(obs_prob))
                prior_prob = tf.matmul(obs_prob_d, next_prob)
                backward_score = tf.matmul(prior_prob, next_prob)
                backward_prob = tf.squeeze(backward_score)
                self.backward = tf.scatter_update(self.backward, step+1, self.scale[self.N-2-step] * backward_prob)
            self.backward = tf.assign(self.backward, tf.reverse(self.backward, 0))


    def _posterior(self):
        self.posterior = tf.multiply(self.forward, self.backward)
        marginal = tf.reduce_sum(self.posterior, 1)
        self.posterior = self.posterior / tf.expand_dims(marginal, 1)

    def reestimate_emission(self, x):
        states_marginal = tf.reduce_sum(self.gamma, 0)
        seq_one_hot = tf.one_hot(tf.cast(x, tf.int64), self.O, 1, 0)
        emission_score = tf.matmul(tf.cast(seq_one_hot, tf.float64), self.gamma, transpose_a=True)
        return emission_score / states_marginal

    def reestimate_transition(self, x):
        with tf.name_scope('init_3d_tensor'):
            self.M = tf.Variable(tf.zeros((self.N-1, self.S, self.S), tf.float64))
        with tf.name_scope('3d_tensor_transition'):
            for t in range(self.N-1):
                with tf.name_scope('time_step_%s' % t):
                    tmp0 = tf.matmul(tf.expand_dims(self.forward[t, :], 0), self.T)
                    tmp1 = tf.multiply(tmp0, tf.expand_dims(tf.gather(self.E, x[t+1]), 0))
                    denom = tf.squeeze(tf.matmul(tmp1, tf.expand_dims(self.backward[t+1,:], 1)))
                with tf.name_scope('init_new_transition'):
                    trans_reestimate = tf.Variable(tf.zeros([self.S, self.S], tf.float32))
                for i in range(self.S):
                    with tf.name_scope('state_%s' % i):
                        numer = self.forward[t, i] * self.T[i, :] * tf.gather(self.E, x[t+1]) * self.backward[t+1,:]
                        trans_reestimate = tf.scatter_update(trans_reestimate, i, numer/denom)
                self.M = tf.scatter_update(self.M, t, trans_reestimate)
        with tf.name_scope('smooth_gamma'):
            self.gamma = tf.squeeze(tf.reduce_sum(self.M, 2))
            T_new = tf.reduce_sum(self.M, 0) / tf.expand_dims(tf.reduce_sum(self.gamma, 0), 1)
        with tf.name_scope('new_init_states_prob'):
            T0_new = self.gamma[0, :]
        with tf.name_scope('append_gamma_final_time_step'):
            prod = tf.expand_dims(tf.multiply(self.forward[self.N-1, :], self.backward[self.N-1, :]), 0)
            s=prod/tf.reduce_sum(prod)
            self.gamma=tf.concat([self.gamma, s], 0)
            self.prob_state_1.append(self.gamma[:, 0])
        return T0_new, T_new

    def check_conv(self, newT0, new_transition, new_emission):
        delta_T0 = tf.reduce_max(tf.abs(self.T0 - newT0)) < self.epsilon
        delta_T = tf.reduce_max(tf.abs(self.T - new_transition)) < self.epsilon
        delta_E = tf.reduce_max(tf.abs(self.E - new_emission)) < self.epsilon
        return tf.logical_and(tf.logical_and(delta_T0, delta_T), delta_E)

    def forward_backward(self, obs_prob_seq):
        pass


