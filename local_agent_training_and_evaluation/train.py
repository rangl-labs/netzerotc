#!/usr/bin/env python

import os
import gym

from util import Trainer

env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")

# Train an RL agent on the environment
trainer = Trainer(env)
os.chdir("./saved_models/")

# To train from scratch:
trainer.train_rl(models_to_train=1, episodes_per_model=1000)

# To resume a previously trained model:
# trainer.train_rl(models_to_train=30, episodes_per_model=1000, last_model_number=9)

os.chdir("../")