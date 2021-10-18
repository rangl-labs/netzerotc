import logging

import pandas as pd
import numpy as np
import gym
# import reference_environment_direct_deployment

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Checks

# Create an environment named env
# fmt: off
env = gym.make("reference_environment_direct_deployment:rangl-nztc-v0")
# fmt: on
# Generate a random action and check it has the right length
action = env.action_space.sample()
assert len(action) == 3

# Reset the environment
env.reset()
# Check the to_observation method
assert len(env.observation_space.sample()) == len(env.state.to_observation())
done = False

# fmt: off
action = [1.0, 1.0, 2.0] # [increment in offshore wind capacity GW, increment in blue hydrogen energy TWh, increment in green hydrogen energy TWh]
# fmt: on
while not done:
    # Specify the action. Check the effect of any fixed policy by specifying the action here:
    observation, reward, done, _ = env.step(action)
    logger.debug(f"step_count: {env.state.step_count}")
    logger.debug(f"action: {action}")
    logger.debug(f"observation: {observation}")
    logger.debug(f"reward: {reward}")
    logger.debug(f"done: {done}")
    print()
    # fmt: off
    action = [2.0, 2.0, 5.0] # env.action_space.sample()
    # fmt: on

# logger.debug(f"env.param.IEV_RewardSensitivities: {env.param.IEV_RewardSensitivities}")

rewards_all = np.array(env.state.weightedRewardComponents_all)
OriginalRewardsFixedPolicyExtract = np.ones(np.shape(rewards_all))
OriginalRewardsFixedPolicyExtract = np.array(np.array(pd.read_excel('./compiled_workbook_objects/Pathways to Net Zero - Original - Fixed Policy Test - Rewards Component Extract.xlsx'))[:,-20:],dtype=np.float64).T
differences = rewards_all[:,:4] - OriginalRewardsFixedPolicyExtract
print(differences)
# IEV_Rewards_1YearShifting = np.ones(np.shape(rewards_all))
# # IEV_Rewards_capexDelta = np.ones(np.shape(rewards_all))
# scenario = 1 # 0 for Breeze, 1 for Gale, 2 for Storm; should be adjusted according to the 'action' above in line 33/35 and 46
# # IEV_Rewards_1YearShifting[:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Total capex.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Total opex.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Total revenue.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,3] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - 1-Year Shifting - Carbon tax for uncaptured carbon.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,4] = np.array(np.array(pd.read_excel('./sensitivities/IEV - 1-Year Shifting - Total Jobs.xlsx'))[scenario,2:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - 1-Year Shifting - Total Economic Impact.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Exact 1-Year Shifting No Double-Deployment - Total capex.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Exact 1-Year Shifting No Double-Deployment - Total opex.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Exact 1-Year Shifting No Double-Deployment - Total revenue.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,3] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Exact 1-Year Shifting No Double-Deployment - Carbon tax for uncaptured carbon.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,4] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Exact 1-Year Shifting No Double-Deployment - Total Jobs.xlsx'))[scenario,2:],dtype=np.float64)
# # IEV_Rewards_1YearShifting[:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Exact 1-Year Shifting No Double-Deployment - Total Economic Impact.xlsx'))[scenario,4:],dtype=np.float64)
# IEV_Rewards_1YearShifting[:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Real 1-Year Shifting No Wave+Tidal - Total capex.xlsx'))[scenario,4:],dtype=np.float64)
# IEV_Rewards_1YearShifting[:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Real 1-Year Shifting No Wave+Tidal - Total opex.xlsx'))[scenario,4:],dtype=np.float64)
# IEV_Rewards_1YearShifting[:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Real 1-Year Shifting No Wave+Tidal - Total revenue.xlsx'))[scenario,4:],dtype=np.float64)
# IEV_Rewards_1YearShifting[:,3] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Real 1-Year Shifting No Wave+Tidal - Carbon tax for uncaptured carbon.xlsx'))[scenario,4:],dtype=np.float64)
# IEV_Rewards_1YearShifting[:,4] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Real 1-Year Shifting No Wave+Tidal - Total Jobs.xlsx'))[scenario,2:],dtype=np.float64)
# IEV_Rewards_1YearShifting[:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Real 1-Year Shifting No Wave+Tidal - Total Economic Impact.xlsx'))[scenario,4:],dtype=np.float64)
# # for a 1-year acceleration/double-deployment in 2021, followed by all 1 year normal pace, the last step should implement the original 2050's rewards, so it should be compared to the original ones, without sensitivity ratio etc.
# # for a 2-year acceleration/triple-deployment in 2021, followed by all 1 year normal pace, the last 2 steps should implement the original 2049's & 2050's rewards, so it should be compared to the original ones, without sensitivity ratio etc.
# IEV_Rewards_1YearShifting[-2:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total capex.xlsx'))[scenario,-2:],dtype=np.float64)
# IEV_Rewards_1YearShifting[-2:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total opex.xlsx'))[scenario,-2:],dtype=np.float64)
# IEV_Rewards_1YearShifting[-2:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total revenue.xlsx'))[scenario,-2:],dtype=np.float64)
# IEV_Rewards_1YearShifting[-2:,3] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Carbon tax for uncaptured carbon.xlsx'))[scenario,-2:],dtype=np.float64)
# IEV_Rewards_1YearShifting[-2:,4] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Jobs.xlsx'))[scenario,-2:],dtype=np.float64)
# IEV_Rewards_1YearShifting[-2:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Economic Impact.xlsx'))[scenario,-2:],dtype=np.float64)
# # IEV_Rewards_capexDelta[:,4] = np.array(np.array(pd.read_excel('./sensitivities/IEV - capex+Delta100 - Total Jobs.xlsx'))[scenario,2:],dtype=np.float64)
# # IEV_Rewards_capexDelta[:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - capex+Delta100 - Total Economic Impact.xlsx'))[scenario,4:],dtype=np.float64)
# # IEV_RewardFormula_1YearShifting = np.array(np.array(pd.read_excel('./sensitivities/IEV - Real 1-Year Shifting No Wave+Tidal - Reward ( = Total Economic Impact - Carbon tax).xlsx'))[scenario,4:],dtype=np.float64)


# # up to now, the rewards output by env.py are correct up to year 2048, so the 
# # calculation for the last 2 elements for year 2049 and 2050 needs to be double checked:
# # (Ref: https://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon)
# # (!Update!: after modifying the env.py for derivative w.r.t. capex: 
# # when the first action = [0, 1, 0, 2, 2, 2] is accelerating for 1 year so that 2021 is a double-deployment,
# # the following line should be used to check the first 4 rewards: capex, opex, revenue, emissions only,
# # because now the jobs & economic impact are now calculated by derivatives approximation + sensitivity ratio,
# # but not just the sensitivity ratio as before; however, for capex calculation itself in the updated version of env.py, 
# # the sensitivityYear loop is changed from 
# # for sensitivityYear in np.arange(state.step_count, IEV_year): 
# # to 
# # for sensitivityYear in np.arange(state.IEV_years[scenario], IEV_year):
# # i.e., the sensitivity ratio will be applied to state.IEV_years[scenario] but no longer back to state.step_count,
# # the new capex for the actual year state.step_count won't be correct, but the capex change on year state.IEV_years[scenario]
# # is still correct for multiplying by the derivative to calculate the approximated change of jobs & economic impact.
# # Therefore, the following line will only check if opex, revenue, emissions are correct)
# # (!Update 2!: after adding codes in env.py to map the state.IEV_years' new capex (with year-shifting + accumulation) back to 
# # the state.step_count using the product of sensitivities, the capex should be correct as well, and also the jobs and economic
# # impact should be correct as well, because all following years after 2021 are fixed 1-year normal pace actions, so except for
# # the 1st step, all other steps' capex's change at the state.IEV_years are 0, so jobs and economic impact at the state.IEV_years
# # should have no change due to size sensitivity/derivative w.r.t. capex, and then multiplying by the 1-year shifting sensitivity
# # ratio to map jobs & economic impact back to state.step_count, they should be the same as the saved xlsx files storing the
# # 1-year shifted jobs & economic impact for all steps/years after 2021, which are checked by the following updated line)
# assert np.amax(np.abs(rewards_all[1:-2,:] - IEV_Rewards_1YearShifting[1:-2,:])) < np.finfo(np.float32).eps # env.py uses float32 in calculation, so numpy.float32's epsilon should be used for checking
# print(rewards_all[0:-2,:] - IEV_Rewards_1YearShifting[0:-2,:])

# # # for jobs & economic impact derivatives w.r.t. capex, using fixed 1-year normal pace actions, the jobs & economic impact from
# # # env.py should be the same as the xlsx files from year 2021 to 2030, when total capex change is manually set to 200 in env.py
# # # (the following line will only check the jobs & economic impact for year 2021 to 2030):
# # assert np.amax(np.abs(rewards_all[0:10,-2:] - IEV_Rewards_capexDelta[0:10,-2:])) < np.finfo(np.float32).eps
# # print(rewards_all[0:10,-2:] - IEV_Rewards_capexDelta[0:10,-2:])

# # # for jobs & economic impact derivatives w.r.t. capex, using fixed 1-year normal pace actions, the jobs & economic impact from
# # # env.py should be the same as the xlsx files from year 2031 to 2050, when total capex change is manually set to 300 in env.py
# # # (the following line will only check the jobs & economic impact for year 2031 to 2050): 
# # assert np.amax(np.abs(rewards_all[10:,-2:] - IEV_Rewards_capexDelta[10:,-2:])) < np.finfo(np.float32).eps
# # print(rewards_all[10:,-2:] - IEV_Rewards_capexDelta[10:,-2:])

# # # To show the errors between env.py's output rewards and the results saved in xlsx, store the errors in a variable and show it
# # # in an IDE, e.g., Spyder or VSCode:
# # errors = rewards_all - IEV_Rewards_capexDelta
# # print(errors)
# errors = rewards_all - IEV_Rewards_1YearShifting
# print(errors)


# Plot the episode
# env.plot("fixed_policy_Breeze1YearNormalPace.png")
# env.plot("fixed_policy_Gale1YearNormalPace.png")
# env.plot("fixed_policy_Storm1YearNormalPace.png")
env.plot("fixed_policy_DirectDeployment.png")
env.plot("10models_100episodes_DirectDeployment.png")
env.plot("10models_100episodes_DirectDeployment_MODEL_1.png")
env.plot("fixed_policy_DirectDeploymentCorrelationRandomized_max(N(1,0),0.5).png")
assert Path("fixed_policy.png").is_file()

# Plot the noise:
import matplotlib.pyplot as plt
from pycel import ExcelCompiler
from IPython.display import FileLink
randomizedPrice = []
for yearColumnID in env.param.Pathways2Net0ColumnInds:
    # fmt: off
    randomizedPrice.append(env.param.Pathways2Net0.evaluate('CCUS!' + yearColumnID + '26'))
    # fmt: on
randomizedPrice = np.array(randomizedPrice)
plt.plot(randomizedPrice)
plt.xlabel("time")
plt.ylabel("price or cost (carbon, capex, opex, or other)")
plt.tight_layout()
# plt.savefig('NoiseVisualization_ApplyingToAllYears.png')
plt.savefig('NoiseVisualization_ApplyingToCurrentStepTo2050.png')

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
env = gym.make("reference_environment_direct_deployment:reference-environment-direct-deployment-v0")
env.seed(123)
obs2 = env.reset()
assert obs1 == obs2

# check that the seed can be reverted to None, so reset() gives different noise
#env = gym.make("reference_environment_direct_deployment:reference-environment-direct-deployment-v0")
#env.seed(123)
#env.seed(None)
#obs1 = env.reset()
#obs2 = env.reset()
#assert not obs1 == obs2


