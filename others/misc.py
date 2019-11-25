import tensorflow as tf
import numpy as np
import numpy.random as random

# sess = tf.Session()
# a = tf.placeholder(tf.int32, [None, ])
# shape = tf.shape(a)
# range = tf.range(shape[0], dtype=tf.int32)
# stack = tf.stack([range, a])
# fd = [3, 5, 7, 9]
# print(sess.run(shape, feed_dict={a: fd}))
# print(sess.run(range, feed_dict={a: fd}))
# print(sess.run(stack, feed_dict={a: fd}))

# for action in range(6):
#     print('action:', (action % 3, action//3))

# a = [0]
# a.extend([0,1,2,3])
# print(a)
#
# dic = {'a': 1, 'b': 2}
# print(dic['a', 'b'])

a = np.array([1,2,3,])
b = np.array([1,2,3])
a += b
print(a)