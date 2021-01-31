# -*- coding: utf-8 -*-
"""
Gym environment wrapper class for improved plotting

Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import gym

import numpy as np
import matplotlib.pyplot as plt

class PlotWrapper(gym.Wrapper):
    """
    Wrapper that modifies the plot function to show realised instead of forecast values in the bottom right.
    """

    def __init__(self, env):
        super().__init__(env)
        
    def plot(self, fname):
        state = self.state

        fig, ax = plt.subplots(2, 2)

        # cumulative total cost
        plt.subplot(221)
        plt.plot(np.cumsum(state.rewards_all))
        plt.xlabel("time")
        plt.ylabel("cumulative reward")
        plt.tight_layout()
        # could be expanded to include individual components of the reward

        # generator levels
        plt.subplot(222)
        plt.plot(np.array(state.generator_1_levels_all))
        plt.plot(np.array(state.generator_2_levels_all))
        plt.xlabel("time")
        plt.ylabel("generator levels")
        plt.tight_layout()


        # actions
        plt.subplot(223)
        plt.plot(np.array(state.actions_all))
        plt.xlabel("time")
        plt.ylabel("actions")
        plt.tight_layout()


        # agent predictions
        plt.subplot(224)
        plt.plot(np.diagonal(np.array(state.agent_predictions_all)))
        plt.plot(np.array(state.generator_1_levels_all) + np.array(state.generator_2_levels_all))
        plt.plot(np.array(state.generator_1_levels_all) + np.array(state.generator_2_levels_all) - np.diagonal(np.array(state.agent_predictions_all)))
        plt.xlabel("time")
        plt.ylabel("realised demand and supply, mismatch")
        plt.tight_layout()


        plt.savefig(fname)

