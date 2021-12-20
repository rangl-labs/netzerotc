#!/usr/bin/env python

import gym
import numpy as np


class Evaluate:
    def __init__(self, env):
        self.env = env
        self.param = env.param

    def random_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                action = self.env.action_space.sample()
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        # TODO we should double check that sampling from the observation space is independent from
        # sampling in the environment which happens with fixed seed
        return np.mean(rewards)


env = gym.make(
    # "reference_environment_direct_deployment:reference-environment-direct-deployment-v0"
    "rangl:nztc-dev-v0"
)

evaluate = Evaluate(env)
mean_reward = evaluate.random_agent(seeds=[123456])

print(f"{mean_reward=}")
