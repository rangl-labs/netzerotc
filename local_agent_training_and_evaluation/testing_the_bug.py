#!/usr/bin/env python

import csv
import numpy as np
import gym
from stable_baselines3 import PPO

import reference_environment_direct_deployment

env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")

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

del model

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
