# -*- coding: utf-8 -*-
"""
Evaluates the clairvoyant agent

Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import logging
import csv
import numpy as np
import gym
import pathlib

# rangl provided modules
import reference_environment
from rangl_local_evaluation import util

# own modules
from modules import envwrapper
from modules import plotwrapper
from modules import zeepkist_mpc

class EvaluateClairvoyant:
    """
    Basic structure for evaluating the clairvoyant agents.
    
    Adapted from provided rangl_local_evaluation/util.py to match the structure of 
    evaluation for non-clairvoyant agents.
    """

    def __init__(self, env, agent):
        self.env = env
        self.param = env.param
        self.agent = agent

    def read_seeds(self, fname="test_set_seeds.csv"):
        file = open(fname)
        csv_file = csv.reader(file)
        seeds = []
        for row in csv_file:
            seeds.append(int(row[0]))
        self.seeds = seeds
        return seeds

    def clairvoyant_agent(self, seeds):
        """Run the clairvoyant agent and return mean reward over the specified list of seeds.

        Args:
            seeds (list): sequence of environment seeds to use

        Returns:
            float: mean reward
        """        
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()

            # store the initial generation levels
            initial_action = [self.env.state.generator_1_level, self.env.state.generator_2_level]

            while not self.env.state.is_done():
                # repeat constant action, just in order to get to the end
                self.env.step(initial_action)
            # read realised demand
            realised_demand = np.diagonal(np.array(env.state.agent_predictions_all))
            # optimise the run cost against (clairvoyant) realised demand, pretending to run at t=-1
            min_cost = agent.full_solution([-1] + initial_action + list(realised_demand))
            # collect (negative) cost
            rewards.append(- min_cost)
        return np.mean(rewards)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create the environment, with full-length wrapper
base_env = gym.make("reference_environment:reference-environment-v0")
env = plotwrapper.PlotWrapper(envwrapper.EfficientObsWrapper(base_env, forecast_length=base_env.param.steps_per_episode))

# Initialise MPC agent
agent = zeepkist_mpc.MPC_agent(env)

# create evaluation environment
evaluate = EvaluateClairvoyant(env, agent)
# get path to file with seeds
seedfile = pathlib.Path(__file__).parents[0] / "rangl_local_evaluation/seeds.csv"
seeds = evaluate.read_seeds(fname=seedfile)
# evaluate mean performance on competition seeds
mean_reward = evaluate.clairvoyant_agent(seeds)

print('Full phase mean reward:',mean_reward)


