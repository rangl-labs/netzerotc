#!/usr/bin/env python

import os
import numpy as np
import gym
from stable_baselines3 import PPO

from util import Evaluate

# fmt: off
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# fmt: on
# agent = PPO.load("./noisy_max(N(1,sqrt(0.001)),0.5)/MODEL_9")
# agent = PPO.load("./deterministic/MODEL_9")
# agent = PPO.load("./deterministic_100xOffshoreWindRevenue/MODEL_9")
# agent = PPO.load("./deterministic_100xEmissions/MODEL_9")
# agent = PPO.load("./deterministic_ActionSpaceOffshoreWindOnly/MODEL_9")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.001x0.1)),0.5)_1000episodes/MODEL_9")
# agent = PPO.load("./deterministic_40models1000episodes/MODEL_39")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.001x0.1)),0.5)_40models1000episodes/MODEL_39")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.001x0.1)),0.5)_40models1000episodes/MODEL_1")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.1)x0.001),0.5)_10models1000episodes/MODEL_9")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.1)x0.01),0.5)_10models1000episodes/MODEL_9")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.001x0.1)),0.5)_40models1000episodes/MODEL_9")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.1)x0.02),0.5)_10models1000episodes/MODEL_9")
# agent = PPO.load("./noisy_max(N(1,sqrt(0.0003x0.1)),0.5)_10models1000episodes/MODEL_9")
# agent = PPO.load("./deterministic_100xEmissions_10models1000episodes/MODEL_9")
# agent = PPO.load("./deterministic_10xEmissions_10models1000episodes/MODEL_9")
# agent = PPO.load("./deterministic_reward_plus_jobs_increment_10models1000episodes/MODEL_9")
agent = PPO.load("./deterministic_reward_plus_jobs_increment_40models1000episodes/MODEL_39")


evaluate = Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on

print("Mean reward:", mean_reward)

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


# rewards_all = np.array(env.state.weightedRewardComponents_all)

# os.chdir("../environment/")
# env.plot(
#     "MODEL_9_10models100episodes_DirectDeploymentCorrelationRandomized_max(N(1,0),0.5).png"
# )
# env.plot("MODEL_9_noisy_max(N(1,sqrt(0.001)),0.5)_10models100episodes_eval_on_deterministic.png")
# env.plot("MODEL_9_deterministic_10models100episodes_eval_on_deterministic.png")
# env.plot("MODEL_9_noisy_max(N(1,sqrt(0.001)),0.5)_10models100episodes_eval_on_noisy.png")
# env.plot("MODEL_9_deterministic_10models100episodes_eval_on_noisy.png")
# env.plot("MODEL_9_deterministic_100xOffshoreWindRevenue_10models100episodes_eval_on_deterministic.png")
# env.plot("MODEL_9_deterministic_100xEmissions_10models100episodes_eval_on_deterministic.png")
# env.plot("MODEL_9_deterministic_ActionSpaceOffshoreWindOnly_10models100episodes_eval_on_deterministic.png")
# env.plot("MODEL_9_noisy_max(N(1,sqrt(0.001x0.1)),0.5)_10models1000episodes_eval_on_deterministic.png")
# env.plot("MODEL_9_noisy_max(N(1,sqrt(0.001x0.1)),0.5)_10models1000episodes_eval_on_noisy.png")
# env.plot("MODEL_39_deterministic_40models1000episodes_eval_on_deterministic.png")
# env.plot("MODEL_39_noisy_max(N(1,sqrt(0.001x0.1)),0.5)_40models1000episodes_eval_on_deterministic.png")
# env.plot("MODEL_1_noisy_max(N(1,sqrt(0.001x0.1)),0.5)_40models1000episodes_eval_on_deterministic.png")
# env.plot("MODEL_39_deterministic_40models1000episodes_eval_on_noisy.png")
# env.plot("MODEL_39_noisy_max(N(1,sqrt(0.001x0.1)),0.5)_40models1000episodes_eval_on_noisy.png")
# os.chdir("../local_agent_training_and_evaluation/")

# os.chdir("./noisy_max(N(1,sqrt(0.1)x0.001),0.5)_10models1000episodes/")
# os.chdir("./noisy_max(N(1,sqrt(0.1)x0.01),0.5)_10models1000episodes/")
# os.chdir("./noisy_max(N(1,sqrt(0.001x0.1)),0.5)_40models1000episodes/")
# os.chdir("./noisy_max(N(1,sqrt(0.1)x0.02),0.5)_10models1000episodes/")
# os.chdir("./noisy_max(N(1,sqrt(0.0003x0.1)),0.5)_10models1000episodes/")
# os.chdir("./deterministic_100xEmissions_10models1000episodes/")
# os.chdir("./deterministic_10xEmissions_10models1000episodes/")
# os.chdir("./deterministic_reward_plus_jobs_increment_10models1000episodes/")
os.chdir("./deterministic_reward_plus_jobs_increment_40models1000episodes/")
# env.plot("MODEL_9_eval_on_noisy.png")
env.plot("MODEL_39_eval_on_deterministic.png")
os.chdir("../")





evaluate = Evaluate(env)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.gale_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on

print("Mean reward:", mean_reward)

assert env.state.weightedRewardComponents_all[-1][3] == 0
print(env.state.weightedRewardComponents_all[-1][3])

env.plot("GALE_eval_on_deterministic_reward_plus_jobs_increment.png")



evaluate = Evaluate(env)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.storm_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on

print("Mean reward:", mean_reward)

assert env.state.weightedRewardComponents_all[-1][3] == 0
print(env.state.weightedRewardComponents_all[-1][3])

env.plot("STORM_eval_on_deterministic_reward_plus_jobs_increment.png")



evaluate = Evaluate(env)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.breeze_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on

print("Mean reward:", mean_reward)

assert env.state.weightedRewardComponents_all[-1][3] == 0
print(env.state.weightedRewardComponents_all[-1][3])

env.plot("BREEZE_eval_on_deterministic_reward_plus_jobs_increment.png")

