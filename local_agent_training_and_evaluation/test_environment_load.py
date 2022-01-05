#!/usr/bin/env python

import gym

env = gym.make("rangl:nztc-open-loop-v0")
print(env)

env = gym.make("rangl:nztc-closed-loop-v0")
print(env)
