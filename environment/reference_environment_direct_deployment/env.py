from pathlib import Path
import numpy as np
import gym
from gym import spaces, logger

from gym.utils import seeding
import matplotlib.pyplot as plt
from pycel import ExcelCompiler


class Parameters:
    # (Avoid sampling random variables here: they would not be resampled upon reset())
    # problem-specific parameters

    techs = 3  # number of technologies (Offshore wind power, blue hydrogen, green hydrogen)
    # fmt: off
    reward_types = 6 # capital expenditure (capex), operating expenditure (opex), revenue, carbon emissions, total jobs supported, total economic impact
    steps_per_episode = 20 # number of years in the planning horizon (2031 -> 2050 = 20)
    # fmt: on
    # This 'Pathways to Net Zero' environment manipulates a spreadsheet loaded in memory. The following 20 columns correspond to years 2031 to 2050 in tabs named 'Outputs' and 'CCUS':
    # fmt: off
    pathways2Net0ColumnInds = np.array(['P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI'])
    # fmt: on
    # The following 20 rows correspond to years 2031 to 2050 in tabs named 'BREEZE', 'GALE', and 'STORM':
    pathways2Net0RowInds = np.arange(36, 36 + steps_per_episode)
    # pathways2Net0ColumnInds[state.step_count] and pathways2Net0RowInds[state.step_count] will locate the current year's column / row respectively
    
    # Multiplicative noise is applied to all costs. The parameters of this randomisation are:
    noise_mu = 1.0
    noise_sigma = 0.1  
    noise_clipping = 0.5  # (i.e., costs are reduced by 50% at the most)
    noise_sigma_factor = np.sqrt(0.1) # this factor is applied to make CCUS capex & opex less volatile than other costs  
    # The costs in the Carbon capture utilisation and storage (CCUS) tab to be randomised are capex, opex, and carbon price, with these row numbers:
    pathways2Net0RandomRowInds_CCUS = np.array([23, 24, 26])
    # The costs in the 'Outputs' tab to be randomised are Offshore wind - Devex, Capex, and Opex, Green Hydrogen - Capex, Fixed Opex, and Variable Opex, Blue Hydrogen - price, Gas feedstock price, Capex, Fixed opex, Variable opex, and Natural gas cost, with these row numbers:
    # fmt: off
    pathways2Net0RandomRowInds_Outputs = np.array([148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166])
    # fmt: on
    

class State:
    def __init__(self, seed=None, param=Parameters()):
        np.random.seed(seed=seed)
        self.initialise_state(param)

    def initialise_state(self, param):
        # create local copy of spreadsheet model to be manipulated
        self.pathways2Net0 = param.pathways2Net0

        # create an array of costs for the current year and populate with 2030 costs (column 'O' in 'CCUS' and 'Outputs' tabs):
        self.randomized_costs = np.ones(
            len(param.pathways2Net0RandomRowInds_CCUS)
            + len(param.pathways2Net0RandomRowInds_Outputs)
        )
        for costRowID in np.arange(len(param.pathways2Net0RandomRowInds_CCUS)):
            self.randomized_costs[costRowID] = param.pathways2Net0.evaluate(
                "CCUS!O" + str(param.pathways2Net0RandomRowInds_CCUS[costRowID])
            )
        for costRowID in np.arange(len(param.pathways2Net0RandomRowInds_Outputs)):
            self.randomized_costs[
                len(param.pathways2Net0RandomRowInds_CCUS) + costRowID
            ] = param.pathways2Net0.evaluate(
                "Outputs!O" + str(param.pathways2Net0RandomRowInds_Outputs[costRowID])
            )

        # time variables
        # NOTE: our convention is to update step_count at the beginning of the gym step() function
        self.step_count = -1
        self.steps_per_episode = param.steps_per_episode
        
        # initial jobs supported in 2030
        self.jobs = np.float32(
            110484
        )  
        # variable to record jobs created each year
        self.jobs_increment = np.zeros(1, dtype=np.float32)  # initialized as 0
        # fmt: off
        # initial economic impact in 2030
        self.econoImpact = np.float32(49938.9809739566)
        # initial technology deployments in 2030
        self.deployments = np.array([param.pathways2Net0.evaluate('GALE!P35'), 
                                     param.pathways2Net0.evaluate('GALE!X35'), 
                                     param.pathways2Net0.evaluate('GALE!Y35')], 
                                    dtype=np.float32) 
        # initial CO2 emissions in 2030
        self.emission_amount = np.float32(param.pathways2Net0.evaluate('CCUS!O63')) 
        # fmt: on

        # histories
        self.observations_all = []
        self.actions_all = []
        self.rewards_all = []
        self.weightedRewardComponents_all = []
        self.deployments_all = []
        self.emission_amount_all = []

    def to_observation(self):
        observation = (self.step_count,) + tuple(
            self.randomized_costs
        )  

        return observation

    def is_done(self):
        done = bool(self.step_count >= self.steps_per_episode - 1)
        return done



def record(state, action, reward, weightedRewardComponents):
    state.observations_all.append(state.to_observation())
    state.actions_all.append(action)
    state.rewards_all.append(reward)
    state.weightedRewardComponents_all.append(weightedRewardComponents)
    state.deployments_all.append(state.deployments)
    state.emission_amount_all.append(state.emission_amount)


def observation_space(self):
    obs_low = np.full_like(self.state.to_observation(), 0, dtype=np.float32)
    obs_low[0] = -1  # first entry of obervation is the timestep
    obs_high = np.full_like(self.state.to_observation(), 1e5, dtype=np.float32)
    obs_high[0] = self.param.steps_per_episode  # first entry of obervation is the timestep
    obs_high[5] = 1e6  
    obs_high[7] = 1e6  
    result = spaces.Box(obs_low, obs_high, dtype=np.float32)
    return result


def action_space(self):
    # action specifies yearly increments in offshore wind power, blue hydrogen, and green hydrogen respectively
    # lower limit on increments is zero
    act_low = np.zeros(self.param.techs, dtype=np.float32)
    # upper limits on increments depend on the technology
    act_high = np.float32([11, 25, 24])
    result = spaces.Box(act_low, act_high, dtype=np.float32)
    return result


def apply_action(action, state, param):
    # copy model from state to param
    param.pathways2Net0 = state.pathways2Net0

    # each technology gives rewards of various types (ie costs and revenues)
    # create an array to hold the reward components (aggregated over all technologies):
    weightedRewardComponents = np.zeros(
        param.reward_types
    )  


    # read in the current deployment for offshore wind power
    offshoreWind = param.pathways2Net0.evaluate(
        "GALE!P" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    # add the increment of offshore wind for this timestep (specified by the action), imposing a maximum deployment
    offshoreWind = np.clip(offshoreWind + action[0], offshoreWind, 150)
    
    # similarly for blue and green hydrogen
    blueHydrogen = param.pathways2Net0.evaluate(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    blueHydrogen = np.clip(blueHydrogen + action[1], blueHydrogen, 270)
    greenHydrogen = param.pathways2Net0.evaluate(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    greenHydrogen = np.clip(greenHydrogen + action[2], greenHydrogen, 253)
    
    # record the new deployments in an array
    state.deployments = np.array(
        [offshoreWind, blueHydrogen, greenHydrogen], dtype=np.float32
    )
    
    # evaluate the model cells containing the deployment values for the current timestep (for offshore wind power, blue hydrogen and green hydrogen respectively)
    # this enables the current timestep's deployment values to be entered into the model 
    param.pathways2Net0.evaluate(
        "GALE!P" + str(param.pathways2Net0RowInds[state.step_count])
    )
    param.pathways2Net0.evaluate(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count])
    )
    param.pathways2Net0.evaluate(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count])
    )
    # similarly, evaluate the current timestep's capex, opex, revenue, and emissions values for all technologies
    # fmt: off
    capex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'41')])
    opex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'42')])
    revenue_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'43')])
    # fmt: on
    emissions = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "68"
        )
    )
    
    # enter the deployment values for this timestep into the model
    param.pathways2Net0.set_value(
        "GALE!P" + str(param.pathways2Net0RowInds[state.step_count]), offshoreWind
    )
    param.pathways2Net0.set_value(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count]), blueHydrogen
    )
    param.pathways2Net0.set_value(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count]), greenHydrogen
    )
    # re-evaluate the current timestep's capex, opex, revenue, and emissions values for all technologies
    # fmt: off
    capex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'41')])
    opex_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'42')])
    revenue_all = np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'43')])
    # fmt: on
    # read gross carbon emissions (before CCUS) from model
    state.emission_amount = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "63"
        )
    )
    # read net carbon emissions (after CCUS) from model
    emissions = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "68"
        )
    )
    # calculate the total capex, opex, revenue and emissions
    weightedRewardComponents[0] = np.sum(capex_all)
    weightedRewardComponents[1] = np.sum(opex_all)
    weightedRewardComponents[2] = np.sum(revenue_all)
    weightedRewardComponents[3] = emissions
    weightedRewardComponents[5] = state.econoImpact
    # calculate numer of jobs supported as 0.25 * (capex + opex + 1050) / 0.05:
    weightedRewardComponents[4] = (
        0.25 * (weightedRewardComponents[0] + weightedRewardComponents[1] + 1050) / 0.05
    )
    state.jobs_increment = weightedRewardComponents[-2] - state.jobs
    state.jobs = weightedRewardComponents[-2]
    # calculate reward for this timestep: revenue - (capex + opex + emissions) + timestep * (increment in jobs) 
    reward = (
        weightedRewardComponents[2] - np.sum(weightedRewardComponents[[0, 1, 3]]) + (state.step_count * state.jobs_increment)
    )  
    # copy model from param to state
    state.pathways2Net0 = param.pathways2Net0
    return state, reward, weightedRewardComponents


def verify_constraints(state):
    verify = True
    # Constraint 1: no decrease in jobs in excess of 25,000 per year
    if (
        state.step_count > 1
    ):  # Note: the index of jobs in reward_types is changed from 3 to 4: capex first, then opex, revenue, emissions, jobs, total economic impact
        if (
            state.weightedRewardComponents_all[-1][4]
            - state.weightedRewardComponents_all[-2][4]
            < -25000
        ):
            verify = False
    # Constraint 2: no decrease in jobs in excess of 37,500 per two years
    if (
        state.step_count > 2
    ):  # Previously in the original env.py, the ordering of reward_types is capex first, then opex, revenue, jobs, emissions
        if (
            state.weightedRewardComponents_all[-1][4]
            - state.weightedRewardComponents_all[-3][4]
            < -37500
        ):
            verify = False
    # Constraint 3: amount of deployment possible in a single year should be less than the maximum single-year capex in any scenario
    # which is the total capex from Storm in 2050 = 26390
    if state.step_count > 0:
        if state.weightedRewardComponents_all[-1][0] > 26390:
            verify = False
    return verify


def randomise(state, action, param):
    param.pathways2Net0 = state.pathways2Net0
    # pass

    # Apply multiplicative noise repeatedly (for each step) to [CCUS Capex (£/tonne), CCUS Opex (£/tonne), Carbon price (£/tonne)] in 'CCUS' spreadsheet's row [23,24,26]
    # and [Offshore wind Devex (£/kW), Offshore wind Capex (£/kW), Offshore wind Opex (£/kW), 
    # Hydrogen green Electrolyser Capex (£/kW H2), Hydrogen green Electrolyser Fixed Opex (£/kW H2), Hydrogen green Electrolyser Variable Opex (£/MWh H2), 
    # Blue Hydrogen price (£/MWh), Gas feedstock (£/MWh), 
    # Hydrogen blue Capex (£million/MW), Hydrogen blue Fixed opex (£million/MW/year), Hydrogen blue Variable opex (£million/TWh), Natural gas cost (£million/TWh)] in
    # 'Outputs' spreadsheet's row [148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166]:
    # Again, before setting the new randomized values to the costs/prices, first evaluate rewards components needed: capex, opex, revenue, and emissions
    # fmt: off
    np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'41')])
    np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'42')])
    np.float32([param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.pathways2Net0.evaluate('Outputs!'+param.pathways2Net0ColumnInds[state.step_count]+'43')])
    np.float32(param.pathways2Net0.evaluate('CCUS!'+param.pathways2Net0ColumnInds[state.step_count]+'68'))
    # fmt: on
    # generate Gaussian noise N~(1,0.1):
    rowInds_CCUS = param.pathways2Net0RandomRowInds_CCUS
    rowInds_Outputs = param.pathways2Net0RandomRowInds_Outputs
    # As in https://github.com/rangl-labs/netzerotc/issues/36, CCUS capex & opex ('CCUS' spreadsheet row 23 and 24) 
    # should have smaller standard deviations by multiplying a factor param.noise_sigma_factor which is < 1:
    noise_sigma_CCUS = np.full(len(rowInds_CCUS), param.noise_sigma) * np.array([param.noise_sigma_factor, param.noise_sigma_factor, 1.0])
    # for multiplicative noise, make sure that the prices/costs are not multiplied by a negative number or zero by clipping to param.noise_clipping from the left:
    multiplicativeNoise_CCUS = np.maximum(
        param.noise_clipping,
        np.random.randn(len(rowInds_CCUS)) * noise_sigma_CCUS + param.noise_mu,
    )
    multiplicativeNoise_Outputs = np.maximum(
        param.noise_clipping,
        np.random.randn(len(rowInds_Outputs)) * param.noise_sigma + param.noise_mu,
    )
    year_counter = 0
    for yearColumnID in param.pathways2Net0ColumnInds[state.step_count :]:
        for costRowID in np.arange(len(rowInds_CCUS)):
            currentCost = param.pathways2Net0.evaluate(
                "CCUS!" + yearColumnID + str(rowInds_CCUS[costRowID])
            )
            # if state.step_count == 0:
            #     currentCost = param.pathways2Net0_reset.evaluate("CCUS!" + yearColumnID + str(rowInds_CCUS[costRowID]))
            param.pathways2Net0.set_value(
                "CCUS!" + yearColumnID + str(rowInds_CCUS[costRowID]),
                multiplicativeNoise_CCUS[costRowID] * currentCost,
            )
            if year_counter == 0:
                state.randomized_costs[costRowID] = (
                    multiplicativeNoise_CCUS[costRowID] * currentCost
                )
        for costRowID in np.arange(len(rowInds_Outputs)):
            currentCost = param.pathways2Net0.evaluate(
                "Outputs!" + yearColumnID + str(rowInds_Outputs[costRowID])
            )
            # if state.step_count == 0:
            #     currentCost = param.pathways2Net0_reset.evaluate("Outputs!" + yearColumnID + str(rowInds_Outputs[costRowID]))
            param.pathways2Net0.set_value(
                "Outputs!" + yearColumnID + str(rowInds_Outputs[costRowID]),
                multiplicativeNoise_Outputs[costRowID] * currentCost,
            )
            if year_counter == 0:
                state.randomized_costs[len(rowInds_CCUS) + costRowID] = (
                    multiplicativeNoise_Outputs[costRowID] * currentCost
                )
        # As in https://github.com/rangl-labs/netzerotc/issues/36, correlated costs are:
        # Hydrogen price = blue hydrogen gas feedstock price + 20, i.e., set row 158 = row 159 + 20 in 'Outputs' spreadsheet:
        param.pathways2Net0.set_value(
            "Outputs!" + yearColumnID + "158",
            param.pathways2Net0.evaluate("Outputs!" + yearColumnID + "159") + 20.0,
        )
        # (more correlated costs in https://github.com/rangl-labs/netzerotc/issues/36, if needed:)

        if year_counter == 0:
            state.randomized_costs[
                len(rowInds_CCUS) + 6
            ] = param.pathways2Net0.evaluate("Outputs!" + yearColumnID + "158")
            # (storing more correlated randomized costs to state.randomized_costs, if needed:)

        # proceed to future years, such that only assigning the current state.step_count/year's randomized costs to state.randomized_costs:
        year_counter = year_counter + 1

    state.pathways2Net0 = param.pathways2Net0
    return state




def reset_param(param):
    # There are two objects of the 'Pathways to Net Zero' model's Excel workbook: param.pathways2Net0 and param.pathways2Net0_reset;
    # in param.pathways2Net0, the prices/costs are randomized, whereas param.pathways2Net0_reset is untouched and is used here to 
    # reset the randomized param.pathways2Net0 to its original state with pre-randomized original prices/costs
    spreadsheets = np.array(["GALE", "CCUS", "Outputs"])
    columnInds_BySheets = np.array(
        [
            np.array(["P", "X", "Y"]),
            param.pathways2Net0ColumnInds,
            param.pathways2Net0ColumnInds,
        ]
    )
    rowInds_BySheets = np.array(
        [
            param.pathways2Net0RowInds,
            param.pathways2Net0RandomRowInds_CCUS,
            param.pathways2Net0RandomRowInds_Outputs,
        ]
    )
    for iSheet in np.arange(len(spreadsheets)):
        for iColumn in columnInds_BySheets[iSheet]:
            for iRow in rowInds_BySheets[iSheet]:
                param.pathways2Net0.set_value(
                    spreadsheets[iSheet] + "!" + iColumn + str(iRow),
                    param.pathways2Net0_reset.evaluate(
                        spreadsheets[iSheet] + "!" + iColumn + str(iRow)
                    ),
                )
    return param

def cal_reset_diff(param):
    abs_diff = 0.0
    # Reload the 'Pathways to Net Zero' model's Excel workbook to pathways2Net0_loaded:
    workbooks_dir = Path(__file__).resolve().parent.parent / "compiled_workbook_objects"
    pathways2Net0_loaded = ExcelCompiler.from_file(filename=f"{workbooks_dir}/PathwaysToNetZero_Simplified_Anonymized_Compiled")
    spreadsheets = np.array(["GALE", "CCUS", "Outputs"])
    columnInds_BySheets = np.array(
        [
            np.array(["P", "X", "Y"]),
            param.pathways2Net0ColumnInds,
            param.pathways2Net0ColumnInds,
        ]
    )
    rowInds_BySheets = np.array(
        [
            param.pathways2Net0RowInds,
            param.pathways2Net0RandomRowInds_CCUS,
            param.pathways2Net0RandomRowInds_Outputs,
        ]
    )
    for iSheet in np.arange(len(spreadsheets)):
        for iColumn in columnInds_BySheets[iSheet]:
            for iRow in rowInds_BySheets[iSheet]:
                if param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None and pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None:
                    abs_diff = abs_diff + np.abs(
                        param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) - pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow))
                    )
                else:
                    if param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None:
                        abs_diff = abs_diff + np.abs(param.pathways2Net0.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)))
                    if pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)) != None:
                        abs_diff = abs_diff + np.abs(pathways2Net0_loaded.evaluate(spreadsheets[iSheet] + "!" + iColumn + str(iRow)))
    # If env.reset() works properly, the abs_diff should be 0, which means that the differences between cell values of pathways2Net0_loaded and param.pathways2Net0 should all be 0
    return abs_diff


def plot_episode(state, fname):
    fig, ax = plt.subplots(2, 2)

    # cumulative total rewards and deployment numbers of the 3 techs
    ax1 = plt.subplot(221)
    plt.plot(np.cumsum(state.rewards_all), label='cumulative reward',color='black')
    plt.xlabel("time, avg reward: " + str(np.mean(state.rewards_all)))
    plt.ylabel("cumulative reward")
    plt.legend(loc='upper left', fontsize='xx-small')
    plt.tight_layout()
    # could be expanded to include individual components of the reward

    ax2 = ax1.twinx()
    ax2.plot(np.array(state.deployments_all)[:,0],label="offshore wind")
    ax2.plot(np.array(state.deployments_all)[:,1],label="blue hydrogen")
    ax2.plot(np.array(state.deployments_all)[:,2],label="green hydrogen")
    ax2.plot(np.array(state.emission_amount_all),label="CO2 emissions amount") 
    ax2.set_ylabel("deployments and CO2 emissions")
    plt.legend(loc='lower right',fontsize='xx-small')
    plt.tight_layout()

    # observations
    plt.subplot(222)
    # first 5 elements of observations are step counts and first 4 randomized costs
    plt.plot(np.array(state.observations_all)[:,0], label="step counts", color='black')
    plt.plot(np.array(state.observations_all)[:,1], label="CCS Capex £/tonne")
    plt.plot(np.array(state.observations_all)[:,2], label="CCS Opex £/tonne")
    plt.plot(np.array(state.observations_all)[:,3], label="Carbon price £/tonne")
    plt.plot(np.array(state.observations_all)[:,4], label="Offshore wind Devex £/kW")
    # plt.plot(np.array(state.observations_all)[:,5], label="Offshore wind Capex £/kW")
    plt.xlabel("time")
    plt.ylabel("observations")
    plt.legend(loc='lower right',fontsize='xx-small')
    plt.tight_layout()

    # actions
    plt.subplot(223)
    plt.plot(np.array(state.actions_all)[:,0],label="offshore wind capacity [GW]")
    plt.plot(np.array(state.actions_all)[:,1],label="blue hydrogen energy [TWh]")
    plt.plot(np.array(state.actions_all)[:,2],label="green hydrogen energy [TWh]")    
    plt.xlabel("time")
    plt.ylabel("actions")
    plt.legend(title="increment in",loc='lower right',fontsize='xx-small')
    plt.tight_layout()

    # jobs and increments in jobs
    plt.subplot(224)
    to_plot = np.vstack((np.array(state.weightedRewardComponents_all)[:,4],
                        np.hstack((np.nan,np.diff(np.array(state.weightedRewardComponents_all)[:,4]))))).T    
    plt.plot(to_plot[:,0], label="jobs")
    plt.plot(to_plot[:,1], label="increment in jobs")
    plt.xlabel("time")
    plt.ylabel("jobs and increments")
    plt.legend(loc='lower left', fontsize='xx-small')
    plt.tight_layout()

    plt.savefig(fname)


def score(state):
    value1 = np.sum(state.rewards_all)
    return {"value1": value1}


class GymEnv(gym.Env):
    def __init__(self):
        self.seed()
        self.load_workbooks()
        self.initialise_state()
    
    def load_workbooks(self):
        self.param = Parameters()
        workbooks_dir = Path(__file__).resolve().parent.parent / "compiled_workbook_objects"
        # There are two objects of the 'Pathways to Net Zero' model's Excel workbook: param.pathways2Net0 and param.pathways2Net0_reset;
        # in param.pathways2Net0, the prices/costs are randomized, whereas param.pathways2Net0_reset is untouched and is used in 
        # reset_param(param) to reset the randomized param.pathways2Net0 to its original state with pre-randomized original prices/costs
        self.param.pathways2Net0 = ExcelCompiler.from_file(filename=f"{workbooks_dir}/PathwaysToNetZero_Simplified_Anonymized_Compiled")
        self.param.pathways2Net0_reset = ExcelCompiler.from_file(filename=f"{workbooks_dir}/PathwaysToNetZero_Simplified_Anonymized_Compiled")

    def initialise_state(self):
        self.param = reset_param(self.param)
        self.state = State(seed=self.current_seed, param=self.param)
        self.action_space = action_space(self)
        self.observation_space = observation_space(self)

    def reset(self):
        # self.load_workbooks()
        self.initialise_state()
        observation = self.state.to_observation()
        return observation
    
    def check_reset(self):
        reset_diff = cal_reset_diff(self.param)
        return reset_diff

    def step(self, action):
        self.state.step_count += 1
        self.state = randomise(self.state, action, self.param)
        self.state, reward, weightedRewardComponents = apply_action(action, self.state, self.param)
        observation = self.state.to_observation()
        done = self.state.is_done()
        record(self.state, action, reward, weightedRewardComponents)
        return observation, reward, done, {}

    def seed(self, seed=None):
        self.current_seed = seed

    def score(self):
        if self.state.is_done():
            return score(self.state)
        else:
            return None
        # print('the score to be returned is: ',score(self.state))
        # return score(self.state)

    def plot(self, fname="episode.png"):
        plot_episode(self.state, fname)

    def render(self):
        pass

    def close(self):
        pass
