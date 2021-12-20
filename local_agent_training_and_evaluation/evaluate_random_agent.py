#!/usr/bin/env python

import gym

from util import Evaluate


env = gym.make("rangl:nztc-dev-v0")

evaluate = Evaluate(env)
mean_reward = evaluate.random_agent(seeds=[123456])

print(f"{mean_reward=}")
