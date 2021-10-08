#!/usr/bin/env python

import os
import numpy as np
import gym
from stable_baselines3 import PPO

from util import Evaluate

# env = gym.make("reference_environment:reference-environment-v0")
os.chdir("../environment/") 
env = gym.make("reference_environment_direct_deployment:reference-environment-direct-deployment-v0")
os.chdir("../local_agent_training_and_evaluation/") 
agent = PPO.load("MODEL_9")


evaluate = Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)

print('Mean reward:',mean_reward)

assert env.state.weightedRewardComponents_all[-1][3] == 0
print(env.state.weightedRewardComponents_all[-1][3])

rewards_all = np.array(env.state.weightedRewardComponents_all)

os.chdir("../environment/")
env.plot("MODEL_9_10models_100episodes_DirectDeploymentRandomized_max(noise,0.001).png")
os.chdir("../local_agent_training_and_evaluation/")
