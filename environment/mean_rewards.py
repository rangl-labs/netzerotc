import logging

import pandas as pd
import numpy as np
import gym
import reference_environment

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create an environment named env
env = gym.make("reference_environment:reference-environment-v0")

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

#Storm:  1716016.402636045
#Gale:  1510077.4347026614
#Breeze:  1325536.6184563776
