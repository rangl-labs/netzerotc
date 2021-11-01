# -*- coding: utf-8 -*-
"""
#Created on Sat Oct 23 2021

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
import scipy.io
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
    weightedRewardComponents_all = np.array(env.state.weightedRewardComponents_all)
    capex_all = weightedRewardComponents_all[:,0]
    jobs_all = weightedRewardComponents_all[:,4]
    jobs_1Yincrements = jobs_all[1:] - jobs_all[:-1]
    jobs_2Yincrements = jobs_all[2:] - jobs_all[:-2]
    return env.score()["value1"], capex_all, jobs_1Yincrements, jobs_2Yincrements

reward_0deployment, capex_all_0deployment, jobs_1Yincrements_0deployment, jobs_2Yincrements_0deployment = calc_rewards(np.zeros((20,3)))

rewards = np.zeros(60)
capex_all = np.zeros((60,20))
jobs_1Yincrements = np.zeros((60,19))
jobs_2Yincrements = np.zeros((60,18))
for i in np.arange(60):
    actions_all = np.zeros(60)
    actions_all[i] = 1.0
    actions_all = actions_all.reshape(20,3,order='F')
    rewards[i], capex_all[i], jobs_1Yincrements[i], jobs_2Yincrements[i] = calc_rewards(actions_all)
rewards_reshape = rewards.reshape(20,3,order='F')

derivatives = rewards - reward_0deployment
derivatives_reshape = rewards_reshape - reward_0deployment

scipy.io.savemat('rewards_all.mat',{'rewards':rewards, 'capex_all':capex_all, 'jobs_1Yincrements':jobs_1Yincrements, 'jobs_2Yincrements':jobs_2Yincrements}, appendmat=False, long_field_names=True, oned_as='column')


calc_rewards( np.tile(np.array([150, 270, 252.797394]), (20, 1)) )

actions_all = np.zeros((20,3))
actions_all[-1,0] = 0.347951864937047
calc_rewards(actions_all)

actions_all = np.zeros(60)
actions_all[0:3] = np.array([3.653724679716039, 1.546067995305606, 0.596394371403215])
actions_all = actions_all.reshape(20,3,order='F')
calc_rewards(actions_all)


env_max = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
env_max.reset()
done = False
max_actions_all = np.tile(np.array([150, 270, 252.797394]), (20, 1))
for i in np.arange(len(max_actions_all)):
    _, _, done, _ = env_max.step(max_actions_all[i])
env_max.score()["value1"]

env_max.plot("max_actions.png")


env_opt_sol = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
env_opt_sol.reset()
done = False
opt_actions_all = np.zeros((20,3))
opt_actions_all[-1,0] = 0.347951864937047
for i in np.arange(len(opt_actions_all)):
    _, _, done, _ = env_opt_sol.step(opt_actions_all[i])
env_opt_sol.score()["value1"]

env_opt_sol.plot("opt_sol.png")


env_opt_sol_ignore_end = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
env_opt_sol_ignore_end.reset()
done = False
opt_ignore_end_actions_all = np.zeros(60)
opt_ignore_end_actions_all[0:3] = np.array([3.653724679716039, 1.546067995305606, 0.596394371403215])
opt_ignore_end_actions_all = opt_ignore_end_actions_all.reshape(20,3,order='F')
for i in np.arange(len(opt_ignore_end_actions_all)):
    _, _, done, _ = env_opt_sol_ignore_end.step(opt_ignore_end_actions_all[i])
env_opt_sol_ignore_end.score()["value1"]

env_opt_sol_ignore_end.plot("opt_sol_ignore_end.png")
