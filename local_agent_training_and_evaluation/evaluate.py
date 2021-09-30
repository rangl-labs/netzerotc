#!/usr/bin/env python

import os
import gym
from stable_baselines3 import PPO

from util import Evaluate

# env = gym.make("reference_environment:reference-environment-v0")
os.chdir("../environment/") 
env = gym.make("reference_environment_direct_deployment:reference-environment-direct-deployment-v0")
os.chdir("../local_agent_training_and_evaluation/") 
agent = PPO.load("MODEL_0")


evaluate = Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)

print('Mean reward:',mean_reward)

assert env.state.weightedRewardComponents_all[-1][3] == 0
print(env.state.weightedRewardComponents_all[-1][3])

os.chdir("../environment/") 
env.plot("10models_100episodes_DirectDeployment_MODEL_0.png")
os.chdir("../local_agent_training_and_evaluation/") 
