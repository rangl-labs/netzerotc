#!/usr/bin/env python

import os
import numpy as np
import gym
from stable_baselines3 import PPO

from util import Evaluate

# fmt: off
# env = gym.make("reference_environment:reference-environment-v0")
env = gym.make("reference_environment_direct_deployment:reference-environment-direct-deployment-v0")
# fmt: on
agent = PPO.load("MODEL_9")


evaluate = Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on

print('Mean reward:',mean_reward)

assert env.state.weightedRewardComponents_all[-1][3] == 0
print(env.state.weightedRewardComponents_all[-1][3])


# mean_rewards = []
# for i in np.arange(10):
#     env = gym.make("reference_environment_direct_deployment:reference-environment-direct-deployment-v0")
#     # env.reset()
#     agent = PPO.load("MODEL_" + str(i))
#     evaluate = Evaluate(env, agent)
#     seeds = evaluate.read_seeds(fname="seeds.csv")
#     # mean_rewards.append(evaluate.RL_agent(seeds))
#     print('Mean reward:',evaluate.RL_agent(seeds))
# mean_rewards = np.array(mean_rewards)
# print('Mean rewards:',mean_rewards)


rewards_all = np.array(env.state.weightedRewardComponents_all)

os.chdir("../environment/")
env.plot("MODEL_9_10models_100episodes_DirectDeploymentCorrelationRandomized_max(N(1,0),0.5).png")
os.chdir("../local_agent_training_and_evaluation/")
