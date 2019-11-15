import tensorflow as tf
import numpy as np
from PPO import PPO
import random
from env.LaneChangeEnv import LaneChangeEnv
import sys


sys.stdout = open('log.txt', 'w')

EP_NUM_MAX = 1000
EP_LEN_MAX = 10000
GAMMA = 0.9
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
MODEL_SAVE_INTERVAL = 5
MODEL_DIR = '../model/'

train_dir = '../model/1/'

with tf.Session() as sess:
    env = LaneChangeEnv()
    ppo = PPO(sess)
    all_ep_r = []

    reward_ph = tf.placeholder(tf.float32, shape=())

    reward_summary = tf.summary.scalar('reward/reward', reward_ph)
    writer = tf.summary.FileWriter(train_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=10)

    for ep in range(EP_NUM_MAX):
        print('ep:', ep)
        egoid = 'lane1.' + str(random.randint(1, 5))
        state = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=False, sumoseed=None, randomseed=None)
        state_np = np.asarray(state).flatten()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN_MAX):    # in one episode
            action = ppo.choose_action(state_np)
            #print('action:', action, (action//3, action % 3))

            state, reward, done, info = env.step((action // 3, action % 3))  # need modification
            state_np = np.asarray(state).flatten()
            is_end_episode = done and info['resetFlag']
            if not is_end_episode:
                buffer_s.append(state_np)
                buffer_a.append(action)
                buffer_r.append(reward)
                #buffer_r.append((r+8)/8)    # normalize reward, find to be useful
                #s = s_
                ep_r += reward

            # update ppo
            if (t+1) % BATCH == 0 or (is_end_episode and len(buffer_s) != 0):
                print('t:', t, 'len_buffer_s', len(buffer_s))
                v_s_ = ppo.get_v(state_np)
                discounted_r = []

                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(buffer_s), np.asarray(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update_old_pi(bs, br)
                for i in range(A_UPDATE_STEPS):
                    summary_aloss_eval = ppo.learn_actor(bs, ba)
                    writer.add_summary(summary_aloss_eval, ppo.actor_step)
                for i in range(C_UPDATE_STEPS):
                    summary_closs_eval = ppo.learn_critic(bs, br)
                    writer.add_summary(summary_closs_eval, ppo.critic_step)
                '''
                for i in range(len(summary_multi_steps_dict['actor_loss'])):
                    writer.add_summary(summary_multi_steps_dict['actor_loss'][i], ep * A_UPDATE_STEPS + i)
                for i in range(len(summary_multi_steps_dict['critic_loss'])):
                    writer.add_summary(summary_multi_steps_dict['critic_loss'][i], ep * C_UPDATE_STEPS + i)
                '''
            if is_end_episode:
                # deleted reset here
                break

        writer.add_summary(sess.run(reward_summary, feed_dict={reward_ph: ep_r}), ep)
        if (ep+1) % MODEL_SAVE_INTERVAL == 0:
            saver.save(sess, MODEL_DIR+'model.ckpt', global_step=ep+1)
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print('Ep: %i' % ep, "|Ep_r: %i" % ep_r)
        print('episode ends\n\n')
    writer.close()
