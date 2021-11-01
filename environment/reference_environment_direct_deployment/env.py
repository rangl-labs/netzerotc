# import math
from pathlib import Path

# import pandas as pd
import numpy as np
import gym
from gym import spaces, logger

from gym.utils import seeding
import matplotlib.pyplot as plt
from pycel import ExcelCompiler

# from IPython.display import FileLink


class Parameters:
    # (Avoid sampling random variables here: they would not be resampled upon reset())
    # problem-specific parameters

    techs = 3  # number of technologies
    scenarios = 3  # number of strategies ('scenarios' in the IEV terminology, eg Breeze, Gale, Storm)
    # fmt: off
    reward_types = 6 # capex first, then opex, revenue, emissions, jobs, total economic impact
    steps_per_episode = 20 # number of years in the planning horizon (eg. 2031 -> 2050 = 20)
    # fmt: on
    # Compile the 'Pathways to Net Zero' Excel work book to a Python object:

    # get the path to the current file
    p = Path(__file__)
    # determine the relative path to the workbooks directory
    workbooks = p.resolve().parent.parent / "compiled_workbook_objects"
    # sensitivities = p.resolve().parent.parent / "sensitivities"

    # pathways2Net0 = ExcelCompiler(filename=f"{workbooks}/Pathways to Net Zero - Simplified.xlsx")
    # pathways2Net0.to_file('./compiled_workbook_objects/Pathways to Net Zero - Simplified - Compiled')
    # read the compiled object from hard drive
    # pathways2Net0 = ExcelCompiler.from_file('./compiled_workbook_objects/Pathways to Net Zero - Simplified - Compiled')
    pathways2Net0 = ExcelCompiler.from_file(
        filename=f"{workbooks}/PathwaysToNetZero_Simplified_Anonymized_Compiled"
    )
    # hard code the columns indices corresponding to year 2031 to 2050 in spreadsheets 'Outputs' and 'CCUS' of the above work book:
    # fmt: off
    pathways2Net0ColumnInds = np.array(['P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI'])
    # fmt: on
    # hard code the row indices corresponding to year 2031 to 2050 in spreadsheets 'BREEZE', 'GALE', and 'STORM' of the above work book:
    pathways2Net0RowInds = np.arange(36, 36 + steps_per_episode)
    # such that pathways2Net0ColumnInds[state.step_count] and pathways2Net0RowInds[state.step_count] will give the
    # corresponding row and column in the spreadsheets
    # Rows in spreadsheet 'CCUS' to be randomized:
    pathways2Net0RandomRowInds_CCUS = np.array([23, 24, 26])
    # Rows in spreadsheet 'Outputs' to be (independently) randomized:
    # pathways2Net0RandomRowInds_Outputs = np.array([148, 149, 150, 153, 154, 155, 159, 163, 164, 165, 166])
    # Rows in spreadsheet 'Outputs' to be randomized:
    # fmt: off
    pathways2Net0RandomRowInds_Outputs = np.array([148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166])
    # fmt: on
    # multiplicative noise's mu and sigma, and clipping point:
    noise_mu = 1.0
    noise_sigma = 0.0  # or try 0.1, 0.0, np.sqrt(0.001), 0.02, np.sqrt(0.0003), 0.015, 0.01
    noise_clipping = 0.5  # or try 0.001, 0.1, 0.5 (i.e., original costs are reduced by 50% at the most)
    noise_sigma_factor = np.sqrt(0.1) # as in https://github.com/rangl-labs/netzerotc/issues/36, CCUS capex & opex (CCUS row 23 and 24) should have smaller standard deviations
    
    # eliminate all constraints to extract rewards coefficients for linear programming:
    no_constraints_testing = False # set to False for reinforcement learning; set to True for linear programming coefficients extractions

    # Compile the IEV economic model work book to a Python object (to be implemented after initial testing):

    # IEV_Rewards = np.ones((scenarios, steps_per_episode, reward_types)) # rewards for each scenario in each year in the IEV model, by reward type (capex first, then opex, revenue, ...)
    # # Ref for importing xlsx file: https://stackoverflow.com/a/49815693
    # IEV_Rewards[:,:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total capex.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    # IEV_Rewards[:,:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total opex.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    # IEV_Rewards[:,:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total revenue.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    # IEV_Rewards[:,:,3] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Carbon tax for uncaptured carbon.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    # IEV_Rewards[:,:,4] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Jobs.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    # IEV_Rewards[:,:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Economic Impact.xlsx'))[:,-steps_per_episode:],dtype=np.float64)


param = Parameters()  # parameters singleton


class State:
    def __init__(self, seed=None):
        np.random.seed(seed=seed)
        self.initialise_state()

    def reset(self):
        self.initialise_state()
        return self.to_observation()

    def initialise_state(self):
        # basic variables
        # self.scenarioWeights = np.full(param.scenarios, 1/3) # a non-negative weight for each scenario which determines its weight in the overall strategy
        # self.scenarioYears = np.ones(param.scenarios) # how many years to advance each scenario during this environment time step (0 for no progress; 1 for normal IEV speed; 2 for double IEV speed; etc)
        # derived variables
        # self.actions = np.concatenate((self.scenarioWeights,self.scenarioYears)) # the RL action at each time step consists of the scenario weights and years

        # forecast variables

        # randomized variables (randomized costs or prices)
        self.randomized_costs = np.ones(
            len(param.pathways2Net0RandomRowInds_CCUS)
            + len(param.pathways2Net0RandomRowInds_Outputs)
        )
        # initialize randomized costs by setting them to fixed (non-randomized) 2030's values (column 'O' in 'CCUS' and 'Outputs'):
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
        # self.IEV_years = np.zeros(param.scenarios, dtype=int) # for each scenario, records the latest IEV year that has been implemented
        self.jobs = np.float32(
            110484
        )  # initial jobs of year 2030, extracted from the IEV model spreadsheet for Gale scenario
        self.jobs_increment = np.zeros(1, dtype=np.float32)  # initialized as 0
        # fmt: off
        self.econoImpact = np.float32(49938.9809739566) # initial economic impact of year 2030, extracted from the IEV model spreadsheet for Gale scenario
        self.deployments = np.array([param.pathways2Net0.evaluate('GALE!P35'), 
                                     param.pathways2Net0.evaluate('GALE!X35'), 
                                     param.pathways2Net0.evaluate('GALE!Y35')], 
                                    dtype=np.float32) # initial deployment numbers of 3 techs in 2030 of Gale scenario
        self.emission_amount = np.float32(param.pathways2Net0.evaluate('CCUS!O63')) # initial CO2 emission amount in 2030 of Gale scenario
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
        )  # + (self.jobs,) + (self.jobs_increment,)# + (self.econoImpact,)

        return observation

    def is_done(self):
        done = bool(self.step_count >= param.steps_per_episode - 1)
        return done

    # def set_agent_prediction(self):
    #    self.agent_prediction = np.reshape(self.cost_predictions, self.GVA_predictions, \
    #        self.summerDemand_predictions, self.winterDemand_predictions, -1)


def record(state, action, reward, weightedRewardComponents):
    state.observations_all.append(state.to_observation())
    state.actions_all.append(action)
    state.rewards_all.append(reward)
    state.weightedRewardComponents_all.append(weightedRewardComponents)
    state.deployments_all.append(state.deployments)
    state.emission_amount_all.append(state.emission_amount)
    # state.agent_predictions_all.append(state.agent_prediction)


def observation_space(self):
    obs_low = np.full_like(self.state.to_observation(), 0, dtype=np.float32)
    obs_low[0] = -1  # first entry of obervation is the timestep
    # obs_low[-1] = -37500 # last entry of obervation is the increment in jobs; Constraint 2: no decrease in jobs in excess of 37,500 per two years
    obs_high = np.full_like(self.state.to_observation(), 1e5, dtype=np.float32)
    obs_high[0] = param.steps_per_episode  # first entry of obervation is the timestep
    obs_high[5] = 1e6  # corresponding to 'Outputs' row 149 Offshore wind capex, whose original maximum is about 2648
    obs_high[7] = 1e6  # corresponding to 'Outputs' row 153 Hydrogen green Electrolyser Capex, whose original maximum is about 1028
    # obs_high[-2] = 10 * 139964 # 2nd last entry of obervation is the jobs; 10 times initial jobs in 2020 = 10*139964, large enough
    # obs_high[-1] = 139964 # last entry of obervation is the increment in jobs; jobs should can't be doubled in a year or increased by the number of total jobs in 2020
    result = spaces.Box(obs_low, obs_high, dtype=np.float32)
    return result


def action_space():
    # the actions are [increment in offshore wind capacity GW, increment in blue hydrogen energy TWh, increment in green hydrogen energy TWh]
    # so the lower bound should be zero because the already deployed cannot be reduced and can only be increased
    act_low = np.zeros(param.techs, dtype=np.float32)
    # the upper bound is set to the highest 2050's target among all 3 scenarios; in other word, the action should not increase the deployment by more than the highest target:
    act_high = np.float32(
        [150, 270, 252.797394]
    )  # Storm's 2050 offshore wind, Breeze's 2050 blue hydrogen, Storm's 2050 green hydrogen
    # act_high = np.float32([150.0, 0.0, 0.0])
    result = spaces.Box(act_low, act_high, dtype=np.float32)
    return result


def apply_action(action, state):

    # capex = 0 # this variable will aggregate all (rebased) capital expenditure for this time step
    weightedRewardComponents = np.zeros(
        param.reward_types
    )  # this array will hold all components of reward for this time step
    # IEV_LastRewards = 0 # this variable will aggregate all other rewards for this time step (these rewards are all assumed to be annual rates)

    # calculate the current state.step_count's deployment after action/increments from previous step (corresponding to row param.pathways2Net0RowInds[state.step_count] - 1),
    # but clip it to the highest 2050 target among 3 scenarios in the spreadsheet:
    offshoreWind = param.pathways2Net0.evaluate(
        "GALE!P" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    offshoreWind = np.clip(offshoreWind + action[0], offshoreWind, 150)
    blueHydrogen = param.pathways2Net0.evaluate(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    blueHydrogen = np.clip(blueHydrogen + action[1], blueHydrogen, 270)
    greenHydrogen = param.pathways2Net0.evaluate(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count] - 1)
    )
    greenHydrogen = np.clip(greenHydrogen + action[2], greenHydrogen, 252.797394)
    # after actions of increments and clipping, assign current state.step_count's deployment numbers to state.deployments:
    state.deployments = np.array(
        [offshoreWind, blueHydrogen, greenHydrogen], dtype=np.float32
    )

    # set these newly actioned deployment numbers into the corresponding cells in 'Gale' spreadsheet of the compiled object:
    # Note: the compiled object is essentially graphs with vertices/nodes and edges for cells and their relations (formulae)
    # if a cell contains raw values/numbers, its cell map/address will already exist; but if a cell originally contains
    # formula (referencing to other cells) but not value, its cell map/address does not readily exist,
    # so it has to be evaluated first such that the cell map/address corresponding to the node/vertex can be generated:
    # pycel's error message if not evaluate to "initialize" the cell map/address:
    # AssertionError: Address "GALE!P36" not found in the cell map. Evaluate the address, or an address that references it, to place it in the cell map.
    param.pathways2Net0.evaluate(
        "GALE!P" + str(param.pathways2Net0RowInds[state.step_count])
    )
    param.pathways2Net0.evaluate(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count])
    )
    param.pathways2Net0.evaluate(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count])
    )
    # also, before resetting the current year's deployment values, the capex opex revenue and emissions of the current year
    # have to be evaluated to initialize the cell's map/address:
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
    # again, before resetting the following 3 values, the above all reward components have to be evaluated first,
    # and after the following resetting, the reward components need to be evaluated again to calculate based on the newly
    # reset values:
    param.pathways2Net0.set_value(
        "GALE!P" + str(param.pathways2Net0RowInds[state.step_count]), offshoreWind
    )
    param.pathways2Net0.set_value(
        "GALE!X" + str(param.pathways2Net0RowInds[state.step_count]), blueHydrogen
    )
    param.pathways2Net0.set_value(
        "GALE!Y" + str(param.pathways2Net0RowInds[state.step_count]), greenHydrogen
    )
    # extract current state.step_count's capex, opex, revenue for all 3 techs, and the emission/carbon tax:
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
    # For current testing only, not involving the IEV economic model work book:
    # one can add new rows in the original 'Outputs' spreadsheets of 'Pathways to Net Zero.xlsx' to calculate the sum,
    # and then directly evaluate the single cell to get the sum, which may be faster or slower than this current approach
    # (that is, running param.pathways2Net0.evaluate() for 5 times and then np.sum(), compared to compiling a slighly complicated
    # workbook with a new row to compute the sum, and then running param.pathways2Net0.evaluate() for 1 time)
    # If the IEV economic model work book is needed, then these values have to be evaluated one by one & input to the IEV model
    state.emission_amount = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "63"
        )
    )
    emissions = np.float32(
        param.pathways2Net0.evaluate(
            "CCUS!" + param.pathways2Net0ColumnInds[state.step_count] + "68"
        )
    )
    # calculate the total capxe, opex, revenue
    weightedRewardComponents[0] = np.sum(capex_all)
    weightedRewardComponents[1] = np.sum(opex_all)
    weightedRewardComponents[2] = np.sum(revenue_all)
    weightedRewardComponents[3] = emissions
    # jobs and total economic impact have to be extracted from the IEV model work book; left for later implementation
    # weightedRewardComponents[4] = param.IEV_EconomicModel.evaluate()
    # weightedRewardComponents[5] = param.IEV_EconomicModel.evaluate()
    # currently set the jobs and economic impact to be the previous values (the above should be implemented after testing):
    # weightedRewardComponents[4] = state.jobs
    weightedRewardComponents[5] = state.econoImpact
    # Update: based on the workshop on 20 Sep, the jobs number calculation we discussed was a simple one:
    # 25% of total costs are spent on salaries and the average salary is GBP 50,000.
    # Since the monetary unit of capex, opex, revenue, decom, etc. is millions of GBP in the compiled Excel workbook,
    # the jobs number in each year should be: 0.25 * (capex + opex + decomm) / 0.05, where the decomm is a fixed constant 1050:
    weightedRewardComponents[4] = (
        0.25 * (weightedRewardComponents[0] + weightedRewardComponents[1] + 1050) / 0.05
    )
    state.jobs_increment = weightedRewardComponents[-2] - state.jobs
    state.jobs = weightedRewardComponents[-2]
    # reward = np.sum(weightedRewardComponents) # sum up the weighted reward components
    # reward = weightedRewardComponents # for testing/checking all components separately, using test_reference_environment.py
    # reward = weightedRewardComponents[-1] - weightedRewardComponents[-3] # proposed reward formula: Reward = Total economic impact - emissions
    reward = (
        weightedRewardComponents[2] - np.sum(weightedRewardComponents[[0, 1, 3]]) - 1050
    )  # new reward formula: - (capex + opex + decomm - revenue) - emissions, where oil & gas decomm is a fixed constant 1050/year for all scenarios
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


def randomise(state, action):
    # pass
    # uncomment to apply multiplicative noise to reward sensitivities
    # param.IEV_RewardSensitivities *= 1
    # uncomment to apply random delay to implementation of IEV years
    # action[param.scenarios:] = np.random.default_rng().integers(np.array(action[param.scenarios:]), endpoint = True)

    # Apply multiplicative noise repeatedly (for each step) to carbon price in 'CCUS' spreadsheet row [23,24,26]
    # and 'Outputs' spreadsheet row [148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166]:
    # Again, before setting the new values to the carbon price, first evaluate rewards components needed: capex opex revenue and emissions
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
    # generate Gaussian N~(1,0.1):
    # rowInds_CCUS = np.array([23,24,26])
    rowInds_CCUS = param.pathways2Net0RandomRowInds_CCUS
    # rowInds_Outputs = np.array([148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166])
    # rowInds_Outputs = np.array([148, 149, 150, 153, 154, 155, 159, 163, 164, 165, 166])
    rowInds_Outputs = param.pathways2Net0RandomRowInds_Outputs
    # As in https://github.com/rangl-labs/netzerotc/issues/36, CCUS capex & opex (CCUS row 23 and 24) 
    # should have smaller standard deviations by multiplying a factor param.noise_sigma_factor which is < 1:
    noise_sigma_CCUS = np.full(len(rowInds_CCUS), param.noise_sigma) * np.array([param.noise_sigma_factor, param.noise_sigma_factor, 1.0])
    # for multiplicative noise, make sure that the prices/costs are not multiplied by a negative number or zero:
    multiplicativeNoise_CCUS = np.maximum(
        param.noise_clipping,
        np.random.randn(len(rowInds_CCUS)) * noise_sigma_CCUS + param.noise_mu,
    )
    multiplicativeNoise_Outputs = np.maximum(
        param.noise_clipping,
        np.random.randn(len(rowInds_Outputs)) * param.noise_sigma + param.noise_mu,
    )
    # for yearColumnID in param.pathways2Net0ColumnInds:
    year_counter = 0
    for yearColumnID in param.pathways2Net0ColumnInds[state.step_count :]:
        for costRowID in np.arange(len(rowInds_CCUS)):
            currentCost = param.pathways2Net0.evaluate(
                "CCUS!" + yearColumnID + str(rowInds_CCUS[costRowID])
            )
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
            param.pathways2Net0.set_value(
                "Outputs!" + yearColumnID + str(rowInds_Outputs[costRowID]),
                multiplicativeNoise_Outputs[costRowID] * currentCost,
            )
            if year_counter == 0:
                state.randomized_costs[len(rowInds_CCUS) + costRowID] = (
                    multiplicativeNoise_Outputs[costRowID] * currentCost
                )
        # https://github.com/rangl-labs/netzerotc/issues/36 correlated costs:
        # Hydrogen price = blue hydrogen gas feedstock price + 20, i.e., set row 158 = row 159 + 20 in 'Outputs' spreadsheet:
        param.pathways2Net0.set_value(
            "Outputs!" + yearColumnID + "158",
            param.pathways2Net0.evaluate("Outputs!" + yearColumnID + "159") + 20.0,
        )
        # more correlated costs in https://github.com/rangl-labs/netzerotc/issues/36:

        if year_counter == 0:
            state.randomized_costs[
                len(rowInds_CCUS) + 6
            ] = param.pathways2Net0.evaluate("Outputs!" + yearColumnID + "158")
            # storing more correlated randomized costs to state.randomized_costs:

        # proceed to future years, such that only assigning the current state.step_count/year's randomized costs to state.randomized_costs:
        year_counter = year_counter + 1


# def update_prediction_array(prediction_array):
# prediction_array = prediction_array + 0.1 * np.random.randn(1,len(prediction_array))[0]
# return prediction_array


def reset_param(param):
    # assuming that the xlsx file contains spreadsheets 'GALE_Backup', 'CCUS_Backup', 'Outputs_Backup' which are duplicated from
    # spreadsheets 'GALE', 'CCUS', 'Outputs' before they are filled with actions in deployments or randomized in the costs/prices,
    # such that 'GALE_Backup', 'CCUS_Backup', 'Outputs_Backup' contain the original blank/empty or pre-randomized values
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
                    param.pathways2Net0.evaluate(
                        spreadsheets[iSheet] + "_Backup!" + iColumn + str(iRow)
                    ),
                )
    return param


def plot_episode(state, fname):
    fig, ax = plt.subplots(2, 2)

    # cumulative total rewards
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

    # generator levels
    plt.subplot(222)
    # plt.plot(np.array(state.observations_all)[:,:4]) # first 4 elements of observations are step counts and 3 IEV years
    # plt.plot(np.array(state.observations_all)[:,-2:]) # last 2 elements of observations are jobs and increments in jobs
    plt.plot(
        np.array(state.observations_all)[:, :5]
    )  # first 5 elements of observations are step counts and first 4 randomized costs
    plt.xlabel("time")
    plt.ylabel("observations")
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

    # # deployment numbers
    # plt.subplot(223)
    # plt.plot(np.array(state.deployments_all))
    # plt.xlabel("time")
    # plt.ylabel("deployments")
    # plt.tight_layout()

    # # actions
    # ax1 = plt.subplot(223)
    # ax1.plot(np.array(state.actions_all))
    # ax1.set_xlabel("time")
    # ax1.set_ylabel("actions")
    # plt.tight_layout()

    # ax2 = ax1.twinx()
    # ax2.plot(np.array(state.deployments_all))
    # ax2.set_ylabel("deployments")
    # plt.tight_layout()

    # jobs
    plt.subplot(224)
    to_plot = np.vstack((np.array(state.weightedRewardComponents_all)[:,4],
                        np.hstack((np.nan,np.diff(np.array(state.weightedRewardComponents_all)[:,4]))))).T    
    plt.plot(to_plot[:,0], label="jobs")
    plt.plot(to_plot[:,1], label="increment in jobs")
    plt.xlabel("time")
    plt.ylabel("jobs and increments")
    plt.legend(loc='lower left', fontsize='xx-small')
    plt.tight_layout()

    # # increments in jobs
    # plt.subplot(235)
    # plt.plot(np.diff(np.array(state.weightedRewardComponents_all)[:,4]))
    # plt.xlabel("time")
    # plt.ylabel("increments in jobs")
    # plt.tight_layout()

    # agent predictions
    # plt.subplot(224)
    # plt.plot(np.array(state.agent_predictions_all))
    # plt.xlabel("time")
    # plt.ylabel("predictions")
    # plt.tight_layout()

    plt.savefig(fname)


def score(state):
    value1 = np.sum(state.rewards_all)
    return {"value1": value1}


class GymEnv(gym.Env):
    def __init__(self):
        self.seed()
        self.initialise_state()

    def initialise_state(self):
        self.state = State(seed=self.current_seed)
        self.action_space = action_space()
        self.observation_space = observation_space(self)
        # self.param = param
        self.param = Parameters()
        # In case that loading the serialized .pkl is too slow when creating a new param by Parameters() above:
        # self.param = reset_param(self.param)

    def reset(self):
        self.initialise_state()
        observation = self.state.to_observation()
        return observation

    def step(self, action):
        self.state.step_count += 1
        # self.state.prediction_array = update_prediction_array(
        #    self.state.prediction_array
        # )
        randomise(self.state, action)
        self.state, reward, weightedRewardComponents = apply_action(action, self.state)
        if self.param.no_constraints_testing == False:
            if verify_constraints(self.state) == False:
                reward = -1000
        # self.state.set_agent_prediction()
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
