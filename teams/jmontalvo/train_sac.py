#%% Imports

import gym
import numpy as np
import time
from gym import wrappers
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from util import Trainer, Evaluate, ObservationWrapper

#%% Parameters

params = {'learning_rate': 0.001,
          'seed': 123,
          'gamma': 0.5,
          #'tau': 0.005,
          'buffer_size': 1000000,
          'batch_size': 256}

obs_params = {'look_ahead': 10,
              'shift': 3,
              'finish': 99,
              'v': 10}

EPISODES = 10000

#%% Train

# Initialize environment
env_raw = gym.make("reference_environment:reference-environment-v0")

# Observation wrapper
env = ObservationWrapper(env_raw, obs_params)

trainer = Trainer(env, params)
trainer.train_rl(episodes_per_model=EPISODES)
