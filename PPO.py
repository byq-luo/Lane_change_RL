import tensorflow as tf
import numpy as np


class PPO(object):
    def __init__(self,
                 sess,
                 S_DIM=12,
                 A_NUM=6,
                 A_LR=0.0001,
                 C_LR=0.0002,
                 EPISILON=0.2):
        self.sess = sess
        self.S_DIM = S_DIM
        self.A_NUM = A_NUM
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.EPISILON = EPISILON

        self.actor_step = 0
        self.critic_step = 0
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.summary_dict = {}
        self.summary_merged_eval_multi_steps = []

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu)
            self.v = tf.layers.dense(l2, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.advantage_eval = None
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.summary_closs = tf.summary.scalar('critic_loss', self.closs)
            self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        # with tf.variable_scope('sample_action'):
        #     self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
                pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
                oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )

                ratio = pi_prob / oldpi_prob
                surr = ratio * self.tfadv
            # clipping method, find this is better
            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-self.EPISILON, 1.+self.EPISILON)*self.tfadv))
            self.summary_aloss = tf.summary.scalar('actor_loss', self.aloss)
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())
        

    def update_old_pi(self, s, r):
        self.sess.run(self.update_oldpi_op)
        self.advantage_eval = self.sess.run(self.advantage, feed_dict={self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

    def learn_actor(self, s, a):
        self.actor_step += 1
        to, summary_aloss_eval = self.sess.run([self.atrain_op, self.summary_aloss],
                                               feed_dict={self.tfs: s, self.tfa: a, self.tfadv: self.advantage_eval})
        return summary_aloss_eval

    def learn_critic(self, s, r):
        self.critic_step += 1
        to, summary_closs_eval = self.sess.run([self.ctrain_op, self.summary_closs],
                                               feed_dict={self.tfs: s, self.tfdc_r: r})
        return summary_closs_eval
        '''
        summary_eval_multi_steps = {}
        summary_actor_loss_multi_steps = []
        summary_critic_loss_multi_steps = []
        # update actor
        # clipping method, find this is better (OpenAI's paper)
        for _ in range(A_UPDATE_STEPS):
            self.actor_step += 1
            to, smr_temp = self.sess.run([self.atrain_op, self.summary_dict['actor_loss']],
                                         feed_dict={self.tfs: s, self.tfa: a, self.tfadv: adv})
            summary_actor_loss_multi_steps.append(smr_temp)
        summary_eval_multi_steps['actor_loss'] = summary_actor_loss_multi_steps
        # update critic
        for _ in range(C_UPDATE_STEPS):
            self.critic_step += 1
            to, smr_temp = self.sess.run([self.ctrain_op, self.summary_dict['critic_loss']],
                                         feed_dict={self.tfs: s, self.tfdc_r: r})
            summary_critic_loss_multi_steps.append(smr_temp)
        summary_eval_multi_steps['critic_loss'] = summary_critic_loss_multi_steps
        return summary_eval_multi_steps
        '''

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l1, 6, tf.nn.softmax, trainable=trainable)
            output = tf.identity(a_prob, name='prob')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return output, params

    def choose_action(self, s):
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


