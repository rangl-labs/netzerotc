# -*- coding: utf-8 -*-
"""
Evaulates the trained RL agent (models/MODEL_0.zip)

Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import logging
import numpy as np
import gym
import pathlib
from stable_baselines3 import SAC

# rangl provided modules
import reference_environment
from rangl_local_evaluation import util

# own modules
from modules import envwrapper
from modules import plotwrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def run_episode(env, agent, plot_name = None):
    """Evaluate agent performance on a single episode and optionally plot the results.

    Args:
        env (gym.Env): environment
        agent : agent to evaluate. Must implement agent.predict(observation, deterministic=True) and return (action, _)
        plot_name (string or Path, optional): name of plot to generate. Defaults to None.

    Returns:
        float: total reward for the episode
    """    
    observation = env.reset()
    done = False
    while not done:
        # Specify the action. Check the effect of any fixed policy by specifying the action here:
        # note: using 'deterministic = True' for evaluation fixes the SAC policy
        action, _states = agent.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
    # plot the episode using the modified function (includes realisations at the bottom right)
    if plot_name is not None:
        env.plot(plot_name)
    return np.sum(env.state.rewards_all)

# create the environment, including action/observation adaptations defined in the envwrapper module
base_env = gym.make("reference_environment:reference-environment-v0")
env = plotwrapper.PlotWrapper(envwrapper.ActWrapper(envwrapper.EfficientObsWrapper(base_env, forecast_length=25)))

# Load agent
modelfile = pathlib.Path(__file__).parents[1] / "models/MODEL_0"
agent = SAC.load(modelfile)

# Perform two independent runs
print(f"Generating plots for two sample runs...")
outputdir = pathlib.Path(__file__).parents[1] / "output"
run_episode(env, agent, outputdir / "agent_run_RL_1.png")
run_episode(env, agent, outputdir / "agent_run_RL_2.png")

# collect results over 50 independent runs, display summary statistics
print(f"\nGenerating results for 50 random runs...")
result_list = np.zeros(50)
for i in range(len(result_list)):
    result_list[i] = run_episode(env, agent)

print(f"Summary statistics of 50 runs:")
print(f"Mean: {np.mean(result_list)}")
print(f"Std: {np.std(result_list)}")
print(f"Min: {np.min(result_list)}")
print(f"Max: {np.max(result_list)}")

# evaluate mean performance on competition seeds
evaluate = util.Evaluate(env, agent)
# get path to file with seeds
seedfile = pathlib.Path(__file__).parents[0] / "rangl_local_evaluation/seeds.csv"
seeds = evaluate.read_seeds(fname=seedfile)
mean_reward = evaluate.RL_agent(seeds) # Add your agent to the Evaluate class and call it here e.g. evaluate.my_agent(seeds)

print('\nFull phase mean reward:',mean_reward)


