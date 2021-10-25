# -*- coding: utf-8 -*-
"""
#Created on Sat Oct 23 2021

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy as Sci
import scipy.linalg
import pandas as pd
import logging
import gym
from pathlib import Path

def calc_rewards(actions_all):    # actions_all is 20 years-by-3 techs
    env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
    env.reset()
    done = False
    for i in np.arange(len(actions_all)):
        observation, reward, done, _ = env.step(actions_all[i])
    return env.score()["value1"]

reward_no_deployment = calc_rewards(np.zeros((20,3)))

rewards = np.zeros(60)
for i in np.arange(60):
    actions_all = np.zeros(60)
    actions_all[i] = 1.0
    actions_all = actions_all.reshape(20,3,order='F')
    rewards[i] = calc_rewards(actions_all)
rewards_reshape = rewards.reshape(20,3,order='F')

derivatives = rewards - reward_no_deployment
derivatives_reshape = rewards_reshape - reward_no_deployment

scipy.io.savemat('rewards.mat',{'rewards':rewards}, appendmat=False, long_field_names=True, oned_as='column')

calc_rewards( np.tile(np.array([150, 270, 252.797394]), (20, 1)) )

env_max = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
env_max.reset()
done = False
max_actions_all = np.tile(np.array([150, 270, 252.797394]), (20, 1))
for i in np.arange(len(max_actions_all)):
    _, _, done, _ = env_max.step(max_actions_all[i])
env_max.score()["value1"]

env_max.plot("max_actions.png")
