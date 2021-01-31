#!/usr/bin/env python
import gym
from stable_baselines3 import PPO
from pathlib import Path
from util import Evaluate, plot2, plot3

env = gym.make("reference_environment:reference-environment-v0")

# agent = PPO.load("MODEL_0.zip")
from stable_baselines3 import DDPG

agent = DDPG.load("MODEL_ALPHA_GENERATION.zip")


evaluate = Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
# mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# mean_reward = evaluate.matching_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# mean_reward = evaluate.min_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)


mean_reward = evaluate.transformed_agent(seeds, H=7,transform="Standard")

# ### Plot the last episode
# plot2(env.state, "fixed_policy")
# plot3(env.state, "fixed_policy.png")
#
#
# ## Plot the last episode
# plot2(env.state, "fixed_policy_h=4")
# plot3(env.state, "fixed_policy_h=4.png")

# plot_policy(env.state, "fixed_policy")
print('Mean reward:', mean_reward)
