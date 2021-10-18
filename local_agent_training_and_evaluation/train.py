#!/usr/bin/env python

import gym

from util import Trainer

# fmt: off
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# fmt: on

# Train an RL agent on the environment
trainer = Trainer(env)
trainer.train_rl(models_to_train=10, episodes_per_model=100)
