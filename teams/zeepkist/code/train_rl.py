# -*- coding: utf-8 -*-
"""
Trains the zeepkist_rl agent (v1.1.1)

Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import logging
import time
import numpy as np
import gym
import pathlib

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

# rangl provided modules
import reference_environment

# own modules
from modules import envwrapper
from modules import plotwrapper

def train(env, models_to_train=1, episodes_per_model=100, **kwargs):
    """
    RL training function. 
    """

    # get path to models directory
    outputpath = pathlib.Path(__file__).parents[1] / "models"

    # using SAC - adjusted gamma to a lower value due to the relatively fast response of the system
    model = SAC(MlpPolicy, env, **kwargs)
    start = time.time()

    for i in range(models_to_train):
        steps_per_model = episodes_per_model * env.param.steps_per_episode
        model.learn(total_timesteps=steps_per_model)
        model.save(outputpath / ("MODEL_" + str(i)))

    end = time.time()
    print("time (min): ", (end - start) / 60)

    # return final model
    return model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create the environment, including action/observation adaptations defined in the envwrapper module
base_env = gym.make("reference_environment:reference-environment-v0")
env = plotwrapper.PlotWrapper(envwrapper.ActWrapper(envwrapper.EfficientObsWrapper(base_env, forecast_length=25)))

# set a default seed for reproducible training
np.random.seed(987654321)

# Train an agent on the environment
agent = train(env, episodes_per_model=5000, verbose=1, gamma=0.85)