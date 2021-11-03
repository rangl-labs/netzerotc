#!/usr/bin/env python

import os
import gym

from util import Trainer

# fmt: off
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# fmt: on

# Train an RL agent on the environment
trainer = Trainer(env)
os.chdir("./deterministic_reward_plus_jobs_increment_40models1000episodes/")
trainer.train_rl(models_to_train=40, episodes_per_model=1000)
os.chdir("../")