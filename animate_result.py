import sys
import os
import random
from LaneChangeEnv import LaneChangeEnv
import tensorflow as tf
import numpy as np

MODEL_DIR = '../model/'
EP_MAX = 10
EP_LEN_MAX = 1000

env = LaneChangeEnv()

with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(MODEL_DIR)
    saver = tf.train.import_meta_graph(latest_checkpoint+'.meta')
    saver.restore(sess, latest_checkpoint)
    print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))

    graph = tf.get_default_graph()
    state_tensor = graph.get_tensor_by_name('state:0')
    prob = graph.get_tensor_by_name('prob:0')

    for ep in range(EP_MAX):
        egoid = 'lane1.' + str(random.randint(1, 5))
        state = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=True, sumoseed=None, randomseed=None)
        state_np = np.asarray(state).flatten()
        for t in range(EP_LEN_MAX):
            prob_eval = sess.run(prob, feed_dict={state_tensor: state_np})
            action = np.random.choice(range(prob_eval.shape[1]), p=prob_eval.ravel())  # select action w.r.t prob

            state, reward, done, info = env.step((action // 3, action % 3))  # need modification
            state_np = np.asarray(state).flatten()

            is_end_episode = done and info['resetFlag']
            if is_end_episode:
                break
