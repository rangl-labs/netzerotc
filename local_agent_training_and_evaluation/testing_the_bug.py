#!/usr/bin/env python

import os
import csv
import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from util import Evaluate
import reference_environment_direct_deployment

env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
"""
### Evaluate RL agents model_0 to model_39
trained_models = 40
episodes_per_model = 1000
mean_rewards = []
for i in [39]:
    agent = PPO.load("MODEL_"+str(i))
    evaluate = Evaluate(env, agent)
    seeds = evaluate.read_seeds(fname="seeds.csv")
    mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
    mean_rewards.append(mean_reward)
print("Mean rewards for RL agent 39:", mean_rewards)

mean_rewards = []
for i in [39]:
    agent = PPO.load("MODEL_"+str(i))
    env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
    #agent.set_env(env)
    evaluate = Evaluate(env, agent)
    seeds = evaluate.read_seeds(fname="seeds.csv")
    mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
    mean_rewards.append(mean_reward)
print("Mean rewards for RL agent 39 again:", mean_rewards)
"""

list1 = []
rewards = []
model = PPO.load("MODEL_0")
seeds = []
for row in csv.reader(open("seeds.csv")):
    seeds.append(int(row[0]))
seeds = seeds[:1]  # try with a single seed to make it fast


for seed in seeds:
    list1.append(seed)
    env.seed(seed)
    obs = env.reset()
    list1.append(obs)
    while not env.state.is_done():
        action, _states = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        list1.append(obs)
        list1.append(action)
rewards.append(sum(env.state.rewards_all))
mean_reward1 = np.mean(rewards)

print("######################################")

# del env
del model
# env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
model = PPO.load("MODEL_0")
list2 = []
rewards = []
for seed in seeds:
    list2.append(seed)
    env.seed(seed)
    obs = env.reset()
    list2.append(obs)
    while not env.state.is_done():
        action, _states = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        list2.append(obs)
        list2.append(action)
rewards.append(sum(env.state.rewards_all))
mean_reward2 = np.mean(rewards)

print("Final results from 1st run: ", mean_reward1)
print("Final results from 2st run: ", mean_reward2)

# for i in range(len(list1)):
#     #if list1[i] != list2[i]:
#     print(i,list1[i],list2[i])
#
# print(list1[-2] == list2[-2])
