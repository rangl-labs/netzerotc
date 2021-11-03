import os
import csv
import pandas as pd
import numpy as np
import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


os.chdir("../environment/")
env = gym.make(
    "reference_environment_direct_deployment:reference-environment-direct-deployment-v0"
)
os.chdir("../local_agent_training_and_evaluation/")

agent = PPO.load("MODEL_1")  # PPO.load("MODEL_0") up to PPO.load("MODEL_9")

seeds = []
for row in csv.reader(open("seeds.csv")):
    seeds.append(int(row[0]))
rewards = []
rewards_all = []
for seed in seeds:
    env.seed(seed)
    obs = env.reset()
    while not env.state.is_done():
        action, _states = agent.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
    rewards.append(sum(env.state.rewards_all))
    rewards_all.append(env.state.rewards_all)
mean_reward = np.mean(rewards)

print("Mean reward:", mean_reward)

plt.plot(np.cumsum(rewards_all[0]))
plt.xlabel("time, avg reward: " + str(np.mean(rewards_all[0])))
plt.ylabel("cumulative reward")
plt.tight_layout()
