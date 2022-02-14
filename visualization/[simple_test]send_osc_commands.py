import logging

import pandas as pd
import numpy as np
import gym
import argparse
import random
import time

from pythonosc import udp_client

import reference_environment_direct_deployment

from pathlib import Path

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Create an environment named env
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")

# Reset the environment
env.reset()

# Setup the OSC
ip = "127.0.0.1"
port = 5005
client = udp_client.SimpleUDPClient(ip, port)  # Create OSC client

rewards = []
deployments = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(env.param.steps_per_episode+1):,1:],dtype=np.float32)
actions = deployments[1:,:] - deployments[:-1,:]
while not env.state.is_done():
    action = actions[env.state.step_count + 1]
    observation, reward, done, _ = env.step(action)
    print(np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount)))
    client.send_message("/some/address", np.hstack((env.state.step_count, reward, env.state.deployments, env.state.emission_amount)))
    # client.send_message("/some/address", env.render())
    

    


def storm_agent(self, seeds):
    rewards = []
    # deployments = np.array(np.array(pd.read_excel('STORM_Deployments.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
    deployments = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
    actions = deployments[1:,:] - deployments[:-1,:]
    for seed in seeds:
        self.env.seed(seed)
        self.env.reset()
        while not self.env.state.is_done():
            action = actions[self.env.state.step_count + 1]
            self.env.step(action)
        rewards.append(sum(self.env.state.rewards_all))
    return np.mean(rewards)
