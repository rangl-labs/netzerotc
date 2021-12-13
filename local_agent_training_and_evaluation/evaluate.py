#!/usr/bin/env python

import os
import numpy as np
import gym
from stable_baselines3 import PPO

from util import Evaluate

env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")

trained_models_dir = "noise_sigma0.1_reward_plus_step_count_jobs_increment_modified_workbook_noNoiseObs"
model_number_str = "39"
agent = PPO.load("./" + trained_models_dir + "/" + "MODEL_" + model_number_str)
evaluate = Evaluate(env, agent)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on
print("Mean reward of model " + model_number_str + ":", mean_reward)

os.chdir("./" + trained_models_dir + "/")
env.plot("MODEL_" + model_number_str + "_eval_on_noisy.png")
# env.plot("MODEL_" + model_number_str + "_eval_on_deterministic.png")
os.chdir("../")


# Reset the environment
env.reset()
del evaluate
evaluate = Evaluate(env)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.random_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on
print("Mean reward of random agent:", mean_reward)

os.chdir("./" + trained_models_dir + "/")
env.plot("random_agent_eval_on_noisy.png")
# env.plot("random_agent_eval_on_deterministic.png")
os.chdir("../")


# Reset the environment
env.reset()
del evaluate
evaluate = Evaluate(env)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.breeze_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on
print("Mean reward of BREEZE:", mean_reward)

os.chdir("./" + trained_models_dir + "/")
env.plot("BREEZE_eval_on_noisy.png")
# env.plot("BREEZE_eval_on_deterministic.png")
os.chdir("../")


# Reset the environment
env.reset()
del evaluate
evaluate = Evaluate(env)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.gale_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on
print("Mean reward of GALE:", mean_reward)

os.chdir("./" + trained_models_dir + "/")
env.plot("GALE_eval_on_noisy.png")
# env.plot("GALE_eval_on_deterministic.png")
os.chdir("../")


# Reset the environment
env.reset()
del evaluate
evaluate = Evaluate(env)
seeds = evaluate.read_seeds(fname="seeds.csv")
# fmt: off
mean_reward = evaluate.storm_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)
# fmt: on
print("Mean reward of STORM:", mean_reward)

os.chdir("./" + trained_models_dir + "/")
env.plot("STORM_eval_on_noisy.png")
# env.plot("STORM_eval_on_deterministic.png")
os.chdir("../")

