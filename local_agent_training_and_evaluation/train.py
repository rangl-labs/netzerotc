#!/usr/bin/env python


import gym

from util import Trainer

env = gym.make("reference_environment:reference-environment-v0")

# Train an RL agent on the environment
trainer = Trainer(env)
trainer.train_rl(models_to_train=1, episodes_per_model=1)

