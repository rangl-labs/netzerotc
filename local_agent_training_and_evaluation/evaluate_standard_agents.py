#!/usr/bin/env python
import os
import csv
import pandas as pd
import numpy as np
import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from util import Evaluate

os.chdir("../environment/")
env = gym.make(
    "reference_environment:rangl-nztc-v0"
)
os.chdir("../local_agent_training_and_evaluation/")

# change the path and name of the model as required:
RL_agent = PPO.load("./saved_models/MODEL_39")

env = gym.make(
    "reference_environment:rangl-nztc-v0"
)

seeds = []
for row in csv.reader(open("seeds.csv")):
    seeds.append(int(row[0]))

evaluate = Evaluate(env,agent=RL_agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
mean_reward_random = evaluate.random_agent(seeds=seeds)
mean_reward_breeze = evaluate.breeze_agent(seeds=seeds)
mean_reward_gale = evaluate.gale_agent(seeds=seeds)
mean_reward_storm = evaluate.storm_agent(seeds=seeds)
mean_reward_RL = evaluate.RL_agent(seeds=seeds)
print("Mean reward")
print("random agent:", mean_reward_random)
print("Breeze agent:", mean_reward_breeze)
print("Gale agent:", mean_reward_gale)
print("Storm agent:", mean_reward_storm)
print("RL agent:", mean_reward_RL)


###Plots
seed=seeds[-1]

# random
env.reset()
env.seed(seed)
obs = env.reset()
while not env.state.is_done():
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
env.plot()
plt.savefig("random_agent.png")

# breeze
env.seed(seed)
env.reset()
deployments = np.array(np.array(pd.read_excel('BREEZE_Deployments.xlsx'))[-(env.param.steps_per_episode + 1):, 1:],
                       dtype=np.float32)
actions = deployments[1:, :] - deployments[:-1, :]
while not env.state.is_done():
    action = actions[env.state.step_count + 1]
    env.step(action)
env.plot()
plt.savefig("breeze_agent.png")


# gale
env.seed(seed)
env.reset()
deployments = np.array(np.array(pd.read_excel('GALE_Deployments.xlsx'))[-(env.param.steps_per_episode + 1):, 1:],
                       dtype=np.float32)
actions = deployments[1:, :] - deployments[:-1, :]
while not env.state.is_done():
    action = actions[env.state.step_count + 1]
    env.step(action)
env.plot()
plt.savefig("gale_agent.png")

# storm
env.seed(seed)
env.reset()
deployments = np.array(np.array(pd.read_excel('STORM_Deployments.xlsx'))[-(env.param.steps_per_episode + 1):, 1:],
                       dtype=np.float32)
actions = deployments[1:, :] - deployments[:-1, :]
while not env.state.is_done():
    action = actions[env.state.step_count + 1]
    env.step(action)
env.plot()
plt.savefig("storm_agent.png")


# RL
seed=seeds[-1]
env.seed(seed)
env.reset()
obs = env.reset()
while not env.state.is_done():
    action, _states = RL_agent.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)
env.plot()
plt.savefig("RL_agent.png")





