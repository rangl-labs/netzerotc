#!/usr/bin/env python


import logging
from pathlib import Path

import pandas as pd
import numpy as np
import gym
import rangl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create an environment named env
env = gym.make("rangl:nztc-closed-loop-v0")

# Generate a random action and check it has the right length
action = env.action_space.sample()
assert len(action) == 3

# Reset the environment
env.reset()
# Check the to_observation method
assert len(env.observation_space.sample()) == len(env.state.to_observation())
done = False

# [increment in offshore wind capacity GW, increment in blue hydrogen energy TWh, increment in green hydrogen energy TWh]
action = [1.0, 1.0, 2.0]

while not done:
    # Specify the action. Check the effect of any fixed policy by specifying the action here:
    observation, reward, done, _ = env.step(action)
    logger.debug(f"step_count: {env.state.step_count}")
    logger.debug(f"action: {action}")
    logger.debug(f"observation: {observation}")
    logger.debug(f"reward: {reward}")
    logger.debug(f"done: {done}")
    print()
    # env.action_space.sample()
    action = [2.0, 2.0, 5.0]


rewards_all = np.array(env.state.weightedRewardComponents_all)
OriginalRewardsFixedPolicyExtract = np.ones(np.shape(rewards_all))
OriginalRewardsFixedPolicyExtract = np.array(
    np.array(
        pd.read_excel(
            "./compiled_workbook_objects/Pathways to Net Zero - Original - Fixed Policy Test - Rewards Component Extract.xlsx"
        )
    )[:, -20:],
    dtype=np.float64,
).T
differences = rewards_all[:, :4] - OriginalRewardsFixedPolicyExtract
print(differences)


# Plot the episode
env.plot("fixed_policy.png")
# env.plot("10models_100episodes.png")
# env.plot("10models_100episodes_MODEL_1.png")
assert Path("fixed_policy.png").is_file()

# logger.info(f"observations_all: {env.state.observations_all}")
# logger.info(f"actions_all: {env.state.actions_all}")
# logger.info(f"rewards_all: {env.state.rewards_all}")

assert len(env.state.observations_all) > 0
assert len(env.state.actions_all) > 0
assert len(env.state.rewards_all) > 0
# Check episode has desired number of steps
assert env.state.observations_all[-1][0] == env.param.steps_per_episode - 1

# check that specifying the same seed gives the same noise
env.seed(123)
obs1 = env.reset()
env = gym.make("rangl:nztc-closed-loop-v0")
env.seed(123)
obs2 = env.reset()
assert obs1 == obs2

# check that the seed can be reverted to None, so reset() gives different noise
# env = gym.make("rangl:nztc-closed-loop-v0")
# env.seed(123)
# env.seed(None)
# obs1 = env.reset()
# obs2 = env.reset()
# assert not obs1 == obs2
