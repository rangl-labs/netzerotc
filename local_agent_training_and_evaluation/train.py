#!/usr/bin/env python

import os
import gym

from util import Trainer

env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")

# Train an RL agent on the environment
trainer = Trainer(env)
os.chdir("./noise_sigma0.01_reward_plus_step_count_jobs_increment_40models1000episodes/")
# trainer.train_rl(models_to_train=40, episodes_per_model=1000, last_model_number=39)
trainer.train_rl(models_to_train=40, episodes_per_model=1000)
os.chdir("../")