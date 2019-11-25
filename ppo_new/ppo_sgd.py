from baselines.common import Dataset, explained_variance, fmt_row, zipsame
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from mpi4py import MPI
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from collections import deque
import random
import sys, os

sys.stdout = open('logs/logg.txt', 'w')


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    egoid = 'lane1.' + str(random.randint(1, 6))
    ob = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=False, sumoseed=None, randomseed=None)
    # ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_ret_detail = np.zeros(3)
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_rets_detail = []
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, 'ep_rets_detail': ep_rets_detail}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            # clear episode
            ep_rets = []
            ep_rets_detail = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, info = env.step(ac)
        new = new and info['resetFlag']
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_ret_detail += np.array([info['r_comf'], info['r_effi'], info['r_safety']])
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_rets_detail.append(cur_ep_ret_detail)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_ret_detail = 0
            cur_ep_len = 0
            egoid = 'lane1.' + str(random.randint(1, 6))
            ob = env.reset(egoid=egoid, tlane=0, tfc=2, is_gui=False, sumoseed=None, randomseed=None)
            # ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)(Generalize Advantage Estimation)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          model_dir_base='./tf_model/',
          is_train=True):

    # tensorboard summary writer & model saving path
    i = 1
    while is_train:
        if not os.path.exists(model_dir_base + str(i)):
            model_dir = model_dir_base + str(i)
            os.makedirs(model_dir)
            break
        else:
            i += 1

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    # KL entropy loss
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    # Clip loss
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    # value function loss
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    # define tensorboard summary scalars
    with tf.name_scope('loss'):
        summary_list_loss = [tf.summary.scalar('total_loss', total_loss)]
        for name in loss_names:
            i = loss_names.index(name)
            summary_list_loss.append(tf.summary.scalar('loss_' + name, losses[i]))
        summary_merged_loss = tf.summary.merge(summary_list_loss)
    with tf.name_scope('reward'):
        reward_total_ph = tf.placeholder(tf.float32, shape=())
        reward_comf_ph = tf.placeholder(tf.float32, shape=())
        reward_effi_ph = tf.placeholder(tf.float32, shape=())
        reward_safety_ph = tf.placeholder(tf.float32, shape=())
        summary_list_reward = [tf.summary.scalar('reward_total', reward_total_ph)]
        summary_list_reward.extend([tf.summary.scalar('reward_comf', reward_comf_ph),
                                    tf.summary.scalar('reward_effi', reward_effi_ph),
                                    tf.summary.scalar('reward_safety', reward_safety_ph)])
        summary_merged_reward = tf.summary.merge(summary_list_reward)
    with tf.name_scope('observation'):
        ego_speed_ph = tf.placeholder(tf.float32, shape=())
        ego_latPos_ph = tf.placeholder(tf.float32, shape=())
        ego_acce_ph = tf.placeholder(tf.float32, shape=())
        #dis2origLeader_ph = tf.placeholder(tf.float32, shape=())
        #dis2trgtLeader_ph = tf.placeholder(tf.float32, shape=())
        #obs_ph_list = [ego_speed_ph, ego_latPos_ph, ego_acce_ph, dis2origLeader_ph, dis2trgtLeader_ph]
        obs_ph_list = [ego_speed_ph, ego_latPos_ph, ego_acce_ph]
        #obs_name_list = ['ego_speed', 'ego_latPos', 'ego_acce', 'dis2origLeader', 'dis2trgtLeader']
        obs_name_list = ['ego_speed', 'ego_latPos', 'ego_acce']
        summary_list_obs = [tf.summary.histogram(name, ph) for name, ph in zip(obs_name_list, obs_ph_list)]
        summary_merged_obs = tf.summary.merge(summary_list_obs)
    # with tf.name_scope('action'):
    #     ac_ph = tf.placeholder(tf.int32, shape=())
    #     summary_list_ac = [tf.summary.histogram('longitudinal', tf.floordiv(ac_ph, 3)),
    #                        tf.summary.histogram('lateral', tf.floormod(ac_ph, 3))]
    #     summary_merged_acs = tf.summary.merge(summary_list_ac)

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    if is_train:
        sess = U.get_session()
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=10)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0, max_seconds > 0]) == 1, \
        "Only one time constraint permitted"

    while is_train:
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        print("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), deterministic=pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values

        print("Optimizing...")
        print(fmt_row(13, loss_names))

        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses_batch = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses_batch.append(newlosses)
            print(fmt_row(13, np.mean(losses_batch, axis=0)))

        print("Evaluating losses...")
        losses_batch = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses_batch.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses_batch, axis=0)
        print(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            print("loss_" + name, lossval)

        # write loss summaries
        summary_eval_loss = sess.run(summary_merged_loss, feed_dict={i: d for i, d in zip(losses, meanlosses)})
        summary_writer.add_summary(summary_eval_loss, iters_so_far)
        # write reward summaries
        for ep_ret, ep_ret_detail in zip(seg['ep_rets'], seg['ep_rets_detail']):
            summary_eval_reward = sess.run(summary_merged_reward, feed_dict={reward_total_ph: ep_ret,
                                                                             reward_comf_ph: ep_ret_detail[0],
                                                                             reward_effi_ph: ep_ret_detail[1],
                                                                             reward_safety_ph: ep_ret_detail[2]})
            summary_writer.add_summary(summary_eval_reward, episodes_so_far)
            episodes_so_far += 1
        # write observation and action summaries
        assert len(seg['ac']) == len(seg['ob'])
        for ac, ob in zip(seg['ac'], seg['ob']):
            summary_eval_obs = sess.run(summary_merged_obs, feed_dict={ego_speed_ph: ob[1],
                                                                       ego_latPos_ph: ob[2],
                                                                       ego_acce_ph: ob[3]})
                                                                       #dis2origLeader_ph: ob[4] - ob[0],
                                                                       #dis2trgtLeader_ph: ob[12] - ob[0]})
            summary_writer.add_summary(summary_eval_obs, timesteps_so_far)
            #summary_eval_acs = sess.run(summary_merged_acs, feed_dict={ac_ph: ac})
            #summary_writer.add_summary(summary_eval_acs, timesteps_so_far)
            timesteps_so_far += 1

        # todo: investigate MPI
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        iters_so_far += 1

        if iters_so_far % 10 == 0:
            saver.save(sess, model_dir + '/model.ckpt', global_step=iters_so_far)
    return pi


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
