import logging

import pandas as pd
import numpy as np
import gym
import reference_environment

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Checks

# Create an environment named env
env = gym.make("reference_environment:reference-environment-v0")



# To test whether jobs and economic impact are correctly changed/calculated using size sensitivity/derivative w.r.t. capex,
# when all actions are fixed 1 year normal pace, and the actual change of capex are manually set to 200 from 2021 and 300 from
# 2031 in env.py (these changes of capex are applied to the IEV model spreadsheets to obtain the changed jobs and economic
# impact and calculate the approximated derivative w.r.t. capex):

# #### Storm
# action = [0, 0, 1, 1, 1, 1] # Fixed 1 year normal pace Storm: [0, 0, 1, 1, 1, 1]
# # To test whether the action of 1 year shifting at the beginning/1st step will give the correct results saved in the xlsx files:
# # action = [0, 1, 0, 2, 2, 2]
# done = False
# while not done:
#     observation, reward, done, _ = env.step(action)
#     action = [0, 0, 1, 1, 1, 1] # env.action_space.sample()
#
# Storm_score = env.score()['value1']
#
# ### Gale
# env.reset()
# action = [0, 1, 0, 1, 1, 1] # Fixed 1 year normal pace Gale
# done = False
# while not done:
#     observation, reward, done, _ = env.step(action)
#     action = [0, 1, 0, 1, 1, 1]
#
# Gale_score = env.score()['value1']
#
#
# ### Breeze
# env.reset()
# action = [1, 0, 0, 1, 1, 1] # Fixed 1 year normal pace: Breeze
# done = False
#
# while not done:
#     observation, reward, done, _ = env.step(action)
#     action = [1, 0, 0, 1, 1, 1]
#
# Breeze_score = env.score()['value1']

### general function
def run_epsiode(action):
    env.reset()
    done = False
    while not done:
        observation, reward, done, _ = env.step(action)
    return(env.score()['value1'])

Storm_action = [0, 0, 1, 1, 1, 1]
Gale_action = [0, 1, 0, 1, 1, 1]
Breeze_action = [1, 0, 0, 1, 1, 1]

equal_mix = [1/3, 1/3, 1/3, 1, 1, 1]

Storm_double = [0, 0, 1, 1, 1, 2]
Gale_double = [0, 1, 0, 1, 2, 1]
Breeze_double = [1, 0, 0, 2, 1, 1]

Storm_5x = [0, 0, 1, 1, 1, 5]
Storm_10x = [0, 0, 1, 1, 1, 10]
Storm_20x = [0, 0, 1, 1, 1, 20]
Storm_30x = [0, 0, 1, 1, 1, 30]

print("Scenario: reward")
print("Storm: ", run_epsiode(Storm_action))
print("Gale: ", run_epsiode(Gale_action))
print("Breeze: ", run_epsiode(Breeze_action))
print("equal weights (0.33): ",run_epsiode(equal_mix))
print("Storm at double speed: ", run_epsiode(Storm_double))
print("Gale at double speed: ", run_epsiode(Gale_double))
print("Breeze at double speed: ", run_epsiode(Breeze_double))
print("Storm at 5 times the normal speed: ", run_epsiode(Storm_5x))
print("Storm at 10 times the normal speed: ", run_epsiode(Storm_10x))
print("Storm at 20 times the normal speed: ", run_epsiode(Storm_20x))
print("Storm at 30 times the normal speed: ", run_epsiode(Storm_30x))
### Results
# print("Scenario: reward")
# print("Storm: ", Storm_score)
# print("Gale: ", Gale_score)
# print("Breeze: ", Breeze_score)

#Storm:  1716016.402636045
#Gale:  1510077.4347026614
#Breeze:  1325536.6184563776
