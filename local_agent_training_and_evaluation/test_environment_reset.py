#!/usr/bin/env python

import csv
import numpy as np
import gym
from stable_baselines3 import PPO

import reference_environment_direct_deployment

# this test shows that we can run the environment twice and get the same mean reward
# the environment is reset using a call to env.reset()

# create the environment
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")

# run the environment for the first time
list1 = []
rewards = []
model = PPO.load("./saved_models/MODEL_0")
seeds = []
for row in csv.reader(open("seeds.csv")):
    seeds.append(int(row[0]))
seeds = seeds[:1]  # try with a single seed to make it fast

for seed in seeds:
    print(f"{seed=}")
    list1.append(seed)
    env.seed(seed)
    obs = env.reset()
    list1.append(obs)

    print(f"INFO: 1st run")
    step = -1
    while not env.state.is_done():
        step += 1
        print(f"INFO: {step=}")
        action, _states = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        list1.append(obs)
        list1.append(action)
rewards.append(sum(env.state.rewards_all))
mean_reward1 = np.mean(rewards)

# for completeness we can delete and reload the model
# though, this is redundant and does not change the outcome
del model
model = PPO.load("./saved_models/MODEL_0")

# run the environment for the second time
list2 = []
rewards = []
for seed in seeds:
    print(f"{seed=}")
    list2.append(seed)
    env.seed(seed)
    obs = env.reset()
    list2.append(obs)

    print(f"INFO: 2nd run")
    step = -1
    while not env.state.is_done():
        step += 1
        print(f"INFO: {step=}")
        action, _states = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        list2.append(obs)
        list2.append(action)
rewards.append(sum(env.state.rewards_all))
mean_reward2 = np.mean(rewards)

# check that we get the same results from both runs
print("Final results from 1st run: ", mean_reward1)
print("Final results from 2nd run: ", mean_reward2)
assert np.isclose(a=[mean_reward1], b=[mean_reward2])
