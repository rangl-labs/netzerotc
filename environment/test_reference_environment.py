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
# Generate a random action and check it has the right length
action = env.action_space.sample()
assert len(action) == 6

# Reset the environment
env.reset()
# Check the to_observation method
assert len(env.observation_space.sample()) == len(env.state.to_observation())
done = False

rewards_all = []
action = [1, 0, 0, 2, 2, 2]
while not done:
    # Specify the action. Check the effect of any fixed policy by specifying the action here:
    observation, reward, done, _ = env.step(action)
    rewards_all.append(reward)
    logger.debug(f"step_count: {env.state.step_count}")
    logger.debug(f"action: {action}")
    logger.debug(f"observation: {observation}")
    logger.debug(f"reward: {reward}")
    logger.debug(f"done: {done}")
    print()
    action = [1, 0, 0, 1, 1, 1] # env.action_space.sample()

logger.debug(f"env.param.IEV_RewardSensitivities: {env.param.IEV_RewardSensitivities}")

rewards_all = np.array(rewards_all)
IEV_Rewards_1YearShifting = np.ones(np.shape(rewards_all))
scenario = 0 # 0 for Breeze, 1 for Gale, 2 for Storm; should be adjusted according to the 'action' above in line 29 and 40
IEV_Rewards_1YearShifting[:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Total capex.xlsx'))[scenario,4:],dtype=np.float64)
IEV_Rewards_1YearShifting[:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Total opex.xlsx'))[scenario,4:],dtype=np.float64)
IEV_Rewards_1YearShifting[:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Total revenue.xlsx'))[scenario,4:],dtype=np.float64)
IEV_Rewards_1YearShifting[:,3] = np.array(np.array(pd.read_excel('./sensitivities/IEV - 1-Year Shifting - Total Jobs.xlsx'))[scenario,2:],dtype=np.float64)
IEV_Rewards_1YearShifting[:,4] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Carbon tax for uncaptured carbon.xlsx'))[scenario,4:],dtype=np.float64)
IEV_Rewards_1YearShifting[:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - 1-Year Shifting - Total Economic Impact.xlsx'))[scenario,4:],dtype=np.float64)

# up to now, the rewards output by env.py are correct up to year 2048, so the 
# calculation for the last 2 elements for year 2049 and 2050 needs to be double checked:
# (Ref: https://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon)
assert np.amax(np.abs(rewards_all[0:-2,:] - IEV_Rewards_1YearShifting[0:-2,:])) < np.finfo(np.float32).eps # env.py uses float32 in calculation, so numpy.float32's epsilon should be used for checking

# Plot the episode
env.plot("fixed_policy.png")
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
env = gym.make("reference_environment:reference-environment-v0")
env.seed(123)
obs2 = env.reset()
assert obs1 == obs2

# check that the seed can be reverted to None, so reset() gives different noise
#env = gym.make("reference_environment:reference-environment-v0")
#env.seed(123)
#env.seed(None)
#obs1 = env.reset()
#obs2 = env.reset()
#assert not obs1 == obs2


