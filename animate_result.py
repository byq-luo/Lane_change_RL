import sys
import os
import random
from LaneChangeEnv import LaneChangeEnv
import tensorflow as tf

MODEL_DIR = '../model/'
env = LaneChangeEnv()
with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(MODEL_DIR)
    saver = tf.train.import_meta_graph(latest_checkpoint+'.meta')
    saver.restore(sess, latest_checkpoint)
    print([n.name for n in tf.get_default_graph().as_graph_def().node])