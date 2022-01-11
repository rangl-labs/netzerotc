#!/usr/bin/env python

import os
import gym

from util import Trainer

env = gym.make("rangl:nztc-open-loop-v0")

# Train an RL agent on the environment
trainer = Trainer(env)
os.chdir("./saved_models/")
# os.chdir("./noise_sigma0.1_modified_workbook_finalized_env.py_DDPG/")
# To resume a previously trained model:
# trainer.train_rl(models_to_train=80, episodes_per_model=1000, last_model_number=39)
# To train from scratch:

trainer.train_rl(models_to_train=40, episodes_per_model=100)
os.chdir("../")
