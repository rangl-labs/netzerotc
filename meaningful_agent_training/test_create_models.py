#!/usr/bin/env python

import os
import gym

from util import Trainer

env = gym.make("rangl:nztc-open-loop-v0")

# Train an RL agent on the environment
trainer = Trainer(env)
os.chdir("./saved_models/")
trainer.train_rl(models_to_train=1, episodes_per_model=100)
os.rename("MODEL_0.zip", "MODEL_open_loop_0.zip")
os.chdir("../")


del env
env = gym.make("rangl:nztc-closed-loop-v0")

# Train an RL agent on the environment
trainer = Trainer(env)
os.chdir("./saved_models/")
trainer.train_rl(models_to_train=1, episodes_per_model=100)
os.rename("MODEL_0.zip", "MODEL_closed_loop_0.zip")
os.chdir("../")
