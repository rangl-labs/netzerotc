#!/usr/bin/env python

import os
import gym

from util import Trainer

env = gym.make("rangl:nztc-dev-v0")

# Train an RL agent on the environment
trainer = Trainer(env)
os.chdir("./saved_models")

# To resume a previously trained model:
# trainer.train_rl(models_to_train=80, episodes_per_model=1000, last_model_number=39)

# To train from scratch:
trainer.train_rl(models_to_train=1, episodes_per_model=100)
os.chdir("../")
