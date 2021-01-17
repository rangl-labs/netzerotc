#!/usr/bin/env python

import gym
from stable_baselines3 import PPO

from util import Evaluate

env = gym.make("reference_environment:reference-environment-v0")
agent = PPO.load("MODEL_0")


evaluate = Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)

print('Mean reward:',mean_reward)
