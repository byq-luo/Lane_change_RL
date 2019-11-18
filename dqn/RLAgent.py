import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
from collections import deque
import numpy as np


class RLAgent():
    # change pretrain to False or True
    def __init__(self, sess, pretrain=False, replay_memory_size=2000, learning_rate=0.0001):
        self.gamma = 0.99  # reward discount factor change. original: 0.98
        self.epsilon_action = 0.1  # the noise added to best action
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=replay_memory_size)
        self._build_model()
        self.sess = sess

    # build tensorflow graph
    def _build_model(self):
        self._summary_list = []
        #self.state = {}
        self.state = []
        self._build_state(self.state, 'state')
        self._action = tf.placeholder(tf.float32, (None,), 'action')

        # use Q-function to calculate Q values and action for prediction Q and target Q. original: trainable = True,
        Q_predict = self._Q_function(self.state, self._action, trainable=True, scope='Q_function')
        # self._chosen_action = self._best_action + tf.truncated_normal(shape=tf.shape(self._best_action), mean=0.0, stddev=self.epsilon_action, name='chosen_action')

        # setup graph for next_Q based on (s',a') (Note: set action=None, trainable=False, its parameters are not updated)
        self.next_state = {}
        self._build_state(self.next_state, 'next_state')
        Q_next = self._Q_function(self.next_state, action=None, trainable=False, scope='next_Q_function')
        self._reward = tf.placeholder(tf.float32, (None,), 'reward')
        Q_target = self._reward + tf.constant(self.gamma, name='gamma') * Q_next

        # setup training ops
        loss = tf.reduce_mean(tf.squared_difference(Q_predict, Q_target))
        self._summary_list.append(tf.summary.scalar('loss', loss))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self._train_op = slim.learning.create_train_op(loss, optimizer, clip_gradient_norm=2)
        # gradient limit to [-2,2], too large gradient not acceptable  # change

        # update weights for next_Q_graph
        Q_variable_list = slim.get_model_variables('Q_function')
        next_Q_variable_list = slim.get_model_variables('next_Q_function')
        self._next_Q_variable_update_ops = [next_Q_variable.assign(Q_variable) for (next_Q_variable, Q_variable) in
                                            zip(next_Q_variable_list, Q_variable_list)]
        self._merged_summary = tf.summary.merge(self._summary_list)

    # setup placeholders for state variables
    def _build_state(self, state, scope):
        with tf.variable_scope(scope):
            '''            state['ego_speed'] = tf.placeholder(tf.float32, (None,), 'ego_speed')
            state['ego_LongitudinalPos'] = tf.placeholder(tf.float32, (None,), 'ego_LongitudinalPos')
            state['ego_LateralPos'] = tf.placeholder(tf.float32, (None,), 'ego_LateralPos')

            state['leader_speed'] = tf.placeholder(tf.float32, (None,), 'leader_speed')
            state['leader_LongitudinalPos'] = tf.placeholder(tf.float32, (None,), 'leader_LongitudinalPos')
            state['leader_LateralPos'] = tf.placeholder(tf.float32, (None,), 'leader_LateralPos')

            state['follower_speed'] = tf.placeholder(tf.float32, (None,), 'follower_speed')
            state['follower_LongitudinalPos'] = tf.placeholder(tf.float32, (None,), 'follower_LongitudinalPos')
            state['follower_LateralPos'] = tf.placeholder(tf.float32, (None,), 'follower_LateralPos')

            state['tgtLeader_speed'] = tf.placeholder(tf.float32, (None,), 'tgtLeader_speed')
            state['tgtLeader_LongitudinalPos'] = tf.placeholder(tf.float32, (None,), 'tgtLeader_LongitudinalPos')
            state['tgtLeader_LateralPos'] = tf.placeholder(tf.float32, (None,), 'tgtLeader_LateralPos')

            state['tgFollower_speed'] = tf.placeholder(tf.float32, (None,), 'tgFollower_speed')
            state['tgFollower_LongitudinalPos'] = tf.placeholder(tf.float32, (None,), 'tgFollower_LongitudinalPos')
            state['tgFollower_LateralPos'] = tf.placeholder(tf.float32, (None,), 'tgFollower_LateralPos')
'''
            for i in range(12):
                state.append(tf.placeholder(tf.float32, (None, )))

    # state feature engineering, in similar scaling
    def _get_state_features(self, state):
        features = tf.stack([np.reshape(state, (12, )), ], axis=1, name='stack_state_features')
        return features

    def _Q_function(self, state, action, trainable=True, scope='Q_function'):
        with tf.variable_scope(scope):
            state_features = self._get_state_features(state)  # a 2d tensor
            # terminal_state_features = self._get_termination_state_features(state)   # a 2d tensor
            # gather each state feature to plot them on tensorboard

            Q = self._build_network(state_features, trainable=trainable)

            if trainable:
                for i in range(state_features.get_shape().as_list()[1]):
                    self._summary_list.append(tf.summary.histogram('state_features_%s' % i,  state_features[i]))

                if action is not None:
                    Q_predict = Q[action]

                self._summary_list.append(tf.summary.histogram('action_result', action))
                self._summary_list.append(tf.summary.histogram('Q_all', Q))
                self._summary_list.append(tf.summary.histogram('Q_chosen_action', Q_predict))

                return Q_predict
            else:
                return tf.reduce_max(Q)

    def _build_network(self, state_features, trainable=True, scope='network'):
        with tf.variable_scope(scope):
            layer1 = slim.fully_connected(
                state_features,
                200,
                activation_fn=tf.nn.relu,
                trainable=trainable,
                scope='fc1'
            )
            layer2 = slim.fully_connected(
                layer1,
                200,
                activation_fn=tf.nn.relu,
                trainable=trainable,
                scope='fc2'
            )
            Q = slim.fully_connected(
                layer2,
                2,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer(),
                biases_initializer=tf.zeros_initializer(),
                trainable=trainable,
                scope='output'
            )

            return Q

    def get_best_action(self, state_eval):
        feed_dict = {self.state[key]: [state_eval[key]] for key in
                     state_eval}  # to add a dimension, make it a rank 2 tensor with batch size=1
        return self.sess.run(self._best_action, feed_dict=feed_dict)[0]

    def get_chosen_action(self, state_eval):
        feed_dict = {self.state[key]: [state_eval[key]] for key in state_eval}
        return self.sess.run(self._chosen_action, feed_dict=feed_dict)[0]

    # update the next_Q network parameters. in the next_Q graph. called periodically in training.
    def update_next_Q_variables(self):
        self.sess.run(self._next_Q_variable_update_ops)

    # store the tuple in replay memory
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # experience replay, to update weights for Q network
    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        state_samples, action_samples, reward_samples, next_state_samples = zip(*mini_batch)
        #state_sample_trans = self._state_sample_transform(state_samples)
        #next_state_sample_trans = self._state_sample_transform(next_state_samples)

        feed_dict1 = {self.state[]: state_samples[i] for i in range(12)}
        feed_dict2 = {self._action: action_samples, self._reward: reward_samples}
        feed_dict3 = {self.next_state[key]: next_state_sample_trans[key] for key in next_state_sample_trans}
        feed_dict = {}
        for sgl_dict in (feed_dict1, feed_dict2, feed_dict3):
            feed_dict.update(sgl_dict)
        _, merged_summary_eval = self.sess.run([self._train_op, self._merged_summary], feed_dict=feed_dict)
        return merged_summary_eval

    # transform state_samples [(speeds),(yaw rates),(yaw angles),(acces)] into shape [(a 4 ele. tuple), (), .., ()]
    def _state_sample_transform(self, state_samples):
        state_element_names = [key for key in state_samples[0]]
        state_sample_trans = {}
        for state in state_samples:
            for name in state_element_names:
                if name not in state_sample_trans:
                    state_sample_trans[name] = []
                state_sample_trans[name].append(state[name])
        return state_sample_trans