#!/usr/bin/env python3
from baselines.common import tf_util as U
from baselines import logger
from env.LaneChangeEnv import LaneChangeEnv
from ppo_new import ppo_sgd


def train(num_timesteps, seed, model_path=None):
    from baselines.ppo1 import mlp_policy
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = LaneChangeEnv()

    logger.log("NOTE: reward will be scaled by a factor of 10  in logged stats. Check the monitor for unscaled reward.")
    pi = ppo_sgd.learn(env, policy_fn,
                       max_timesteps=num_timesteps,
                       timesteps_per_actorbatch=512,
                       clip_param=0.1, entcoeff=0.0,
                       optim_epochs=10,
                       optim_stepsize=1e-4,
                       optim_batchsize=64,
                       gamma=0.99,
                       lam=0.95,
                       schedule='constant',
                       )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi

def main():
    logger.configure()

    # train the model
    train(num_timesteps=10000, seed=None, model_path='../model1')

if __name__ == '__main__':
    main()
