#!/usr/bin/env python

import os
import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from util import Evaluate
import reference_environment_direct_deployment

env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")

# Evaluate RL agents model_0 to model_9
trained_models = 10
episodes_per_model = 100
mean_rewards = []
for i in range(trained_models):
    print(f"Loading model {i}")
    agent = PPO.load("MODEL_" + str(i))
    evaluate = Evaluate(env, agent)
    seeds = evaluate.read_seeds(fname="seeds.csv")
    
    # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
    mean_reward = evaluate.RL_agent(seeds)  
    mean_rewards.append(mean_reward)
print("Mean rewards for RL agents:", mean_rewards)

# Plot learning curve
plt.plot(mean_rewards)
plt.xlabel(
    f"training time ({trained_models} models with {episodes_per_model} episodes each)"
)
plt.ylabel("reward")
plt.title("learning curve")
plt.savefig("learning curve.png")
