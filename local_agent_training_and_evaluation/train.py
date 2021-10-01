#!/usr/bin/env python


import os
import gym

from util import Trainer

# env = gym.make("reference_environment:reference-environment-v0")
os.chdir("../environment/") 
env = gym.make("reference_environment_direct_deployment:reference-environment-direct-deployment-v0")
os.chdir("../local_agent_training_and_evaluation/") 

# Train an RL agent on the environment
trainer = Trainer(env)
trainer.train_rl(models_to_train=10, episodes_per_model=100)

