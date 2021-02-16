import logging

import gym
import reference_environment
import numpy as np
from ddpg_agent_training.utils.agent import agent
import argparse
from ddpg_agent_training.utils.normalized_env import NormalizedEnv

# Create an environment

env = gym.make("reference_environment:reference-environment-v0")

#we don't use normalized environment as it pergorms worse
# env = NormalizedEnv(env, ob=True, ret=False)



#environment parameters
env_params = {'obs': env.observation_space.shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high,
            'reward': 1,
            'max_timesteps': 96
            }

args = argparse.Namespace(n_epochs=100, action_l2=1., batch_size=128, buffer_size=100000, \
                          cuda=False, demo_length=20, env_name='reference_environment:reference-environment-v0', \
                          gamma=0.98, lr_actor=0.001, scale_reward=1., clip_range=2, clip_obs= 20,\
                          lr_critic=0.001, n_batches=1, n_cycles=1, n_test_rollouts=1, norm_param=1, \
                          num_rollouts_per_mpi=1, num_workers=1, polyak=0.001, noise_eps=0.1, random_eps=0.3,\
                          save_dir='trained_models/', save_interval=5, seed=123)

ddpg_trainer = agent(args, env, env_params)

ddpg_trainer.learn()



#----------------------Experemental settings-------------------------------
# before epoch 80 it has -2400 average rewards
# args = argparse.Namespace(action_l2=1, batch_size=64, buffer_size=1000000, min_buffer_size=10000, clip_obs=200, clip_range=5, clip_return=50, \
#                           cuda=False, demo_length=20, env_name='reference_environment:reference-environment-v0', \
#                           gamma=0.98, lr_actor=0.001, scale_reward=0.01, ou_theta = 0.15, ou_mu=0., ou_sigma=0.3, exploration_time=32,\
#                           lr_critic=0.001, n_batches=1, n_cycles=1, n_epochs=1000, n_test_rollouts=1, noise_eps=0.2, \
#                           num_rollouts_per_mpi=1, num_workers=1, polyak=0.001, random_eps=0.3, replay_k=4, \
#                           replay_strategy='future', save_dir='saved_models/', save_interval=5, seed=123, skip=40, divis=15)