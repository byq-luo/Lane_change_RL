import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PPO import PPO
import random
from LaneChangeEnv import LaneChangeEnv

EP_NUM_MAX = 1000
EP_LEN_MAX = 10000
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM = 12
A_NUM = 6
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization
train_dir = '../model/'

env = LaneChangeEnv()
ppo = PPO()
all_ep_r = []

with tf.Session() as sess:
    reward_ph = tf.placeholder(tf.float32, shape=())
    #reward_ph = tf.placeholder(tf.float32, shape=())

    reward_summary = tf.summary.scalar('reward/reward', reward_ph)
    writer = tf.summary.FileWriter(train_dir, sess.graph)

    for ep in range(EP_NUM_MAX):

        egoid = 'lane1.' + str(random.randint(1, 5))
        state = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=False, sumoseed=None, randomseed=None)
        state_np = np.asarray(state).flatten()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN_MAX):    # in one episode
            action = ppo.choose_action(state_np)
            print('action:', action, (action//3, action % 3))

            state, reward, done, info = env.step((action // 3, action % 3))  # need modification
            is_end_episode = done and info['resetFlag']
            if not is_end_episode:
                buffer_s.append(state_np)
                buffer_a.append(action)
                buffer_r.append(reward)
            #buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            #s = s_
                ep_r += reward

            # update ppo
            if (t+1) % BATCH == 0 or is_end_episode:
                v_s_ = ppo.get_v(state_np)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(buffer_s), np.asarray(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)

            if is_end_episode:
                state = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=False, sumoseed=None, randomseed=None)
                break
        writer.add_summary(reward_summary, ep)
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )

    writer.close()
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward'); plt.show()