import math

import pandas as pd
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
import matplotlib.pyplot as plt
from pycel import ExcelCompiler
from IPython.display import FileLink


class Parameters:
    # (Avoid sampling random variables here: they would not be resampled upon reset())
    # problem-specific parameters
    
    techs = 3 # number of technologies
    scenarios = 3 # number of strategies ('scenarios' in the IEV terminology, eg Breeze, Gale, Storm)
    reward_types = 6 # capex first, then opex, revenue, emissions, jobs, total economic impact
    steps_per_episode = 20 # number of years in the planning horizon (eg. 2031 -> 2050 = 20)
    # Compile the 'Pathways to Net Zero' Excel work book to a Python object:
    Pathways2Net0 = ExcelCompiler(filename='./compiled_workbook_objects/Pathways to Net Zero - Simplified.xlsx')
    # Pathways2Net0.to_file('./compiled_workbook_objects/Pathways to Net Zero - Simplified - Compiled')
    # read the compiled object from hard drive
    # Pathways2Net0 = ExcelCompiler.from_file('./compiled_workbook_objects/Pathways to Net Zero - Simplified - Compiled')
    # hard code the columns indices corresponding to year 2031 to 2050 in spreadsheets 'Outputs' and 'CCUS' of the above work book:
    Pathways2Net0ColumnInds = np.array(['P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI'])
    # hard code the row indices corresponding to year 2031 to 2050 in spreadsheets 'BREEZE', 'GALE', and 'STORM' of the above work book:
    Pathways2Net0RowInds = np.arange(36,36+steps_per_episode)
    # Pathways2Net0ColumnInds[state.step_count] and Pathways2Net0RowInds[state.step_count] will give the 
    # corresponding row and column in the spreadsheets
    
    # Compile the IEV economic model work book to a Python object (to be implemented after initial testing):
    
    
    IEV_Rewards = np.ones((scenarios, steps_per_episode, reward_types)) # rewards for each scenario in each year in the IEV model, by reward type (capex first, then opex, revenue, ...)
    # Ref for importing xlsx file: https://stackoverflow.com/a/49815693
    IEV_Rewards[:,:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total capex.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    IEV_Rewards[:,:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total opex.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    IEV_Rewards[:,:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total revenue.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    IEV_Rewards[:,:,3] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Carbon tax for uncaptured carbon.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    IEV_Rewards[:,:,4] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Jobs.xlsx'))[:,-steps_per_episode:],dtype=np.float64)
    IEV_Rewards[:,:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Economic Impact.xlsx'))[:,-steps_per_episode:],dtype=np.float64)

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
        #self.scenarioWeights = np.full(param.scenarios, 1/3) # a non-negative weight for each scenario which determines its weight in the overall strategy
        #self.scenarioYears = np.ones(param.scenarios) # how many years to advance each scenario during this environment time step (0 for no progress; 1 for normal IEV speed; 2 for double IEV speed; etc)
        # derived variables
        #self.actions = np.concatenate((self.scenarioWeights,self.scenarioYears)) # the RL action at each time step consists of the scenario weights and years

        # forecast variables

        # time variables
        # NOTE: our convention is to update step_count at the beginning of the gym step() function
        self.step_count = -1
        # self.IEV_years = np.zeros(param.scenarios, dtype=int) # for each scenario, records the latest IEV year that has been implemented
        self.jobs = np.float32(110484) # initial jobs of year 2030, extracted from the IEV model spreadsheet for Gale scenario
        self.jobs_increment = np.zeros(1, dtype=np.float32) # initialized as 0
        self.EconoImpact = np.float32(49938.9809739566) # initial economic impact of year 2030, extracted from the IEV model spreadsheet for Gale scenario
        
        # histories
        self.observations_all = []
        self.actions_all = []
        self.rewards_all = []
        self.weightedRewardComponents_all = [] 

    def to_observation(self):
        observation = (
            self.step_count,
        )# + (self.jobs,) + (self.jobs_increment,)# + (self.EconoImpact,)
        
        return observation

    def is_done(self):
        done = bool(
            self.step_count >= param.steps_per_episode - 1
            )
        return done

    # def set_agent_prediction(self):
    #    self.agent_prediction = np.reshape(self.cost_predictions, self.GVA_predictions, \
    #        self.summerDemand_predictions, self.winterDemand_predictions, -1)
        

def record(state, action, reward, weightedRewardComponents):
    state.observations_all.append(state.to_observation())
    state.actions_all.append(action)
    state.rewards_all.append(reward)
    state.weightedRewardComponents_all.append(weightedRewardComponents)
    # state.agent_predictions_all.append(state.agent_prediction)


def observation_space(self):
    obs_low = np.full_like(self.state.to_observation(), 0, dtype=np.float32)
    obs_low[0] = -1	# first entry of obervation is the timestep
    # obs_low[-1] = -37500 # last entry of obervation is the increment in jobs; Constraint 2: no decrease in jobs in excess of 37,500 per two years
    obs_high = np.full_like(self.state.to_observation(), 1000, dtype=np.float32) 
    obs_high[0] = param.steps_per_episode	# first entry of obervation is the timestep
    # obs_high[-2] = 10 * 139964 # 2nd last entry of obervation is the jobs; 10 times initial jobs in 2020 = 10*139964, large enough
    # obs_high[-1] = 139964 # last entry of obervation is the increment in jobs; jobs should can't be doubled in a year or increased by the number of total jobs in 2020
    result = spaces.Box(obs_low, obs_high, dtype=np.float32)
    return result


def action_space():
    # the actions are [increment in offshore wind capacity GW, increment in blue hydrogen energy TWh, increment in green hydrogen energy TWh]
    # so the lower bound should be zero because the already deployed cannot be reduced and can only be increased
    act_low = np.zeros(param.techs, dtype=np.float32)
    # the upper bound is set to the highest 2050's target among all 3 scenarios; in other word, the action should not increase the deployment by more than the highest target:
    act_high = np.float32([150,270,252.797394]) # Storm's 2050 offshore wind, Breeze's 2050 blue hydrogen, Storm's 2050 green hydrogen
    result = spaces.Box(act_low, act_high, dtype=np.float32)
    return result


def apply_action(action, state):
    
    #capex = 0 # this variable will aggregate all (rebased) capital expenditure for this time step
    weightedRewardComponents = np.zeros(param.reward_types) # this array will hold all components of reward for this time step
    # IEV_LastRewards = 0 # this variable will aggregate all other rewards for this time step (these rewards are all assumed to be annual rates)
    
    # calculate the current state.step_count's deployment after action/increments from previous step (corresponding to row param.Pathways2Net0RowInds[state.step_count] - 1), 
    # but clip it to the highest 2050 target among 3 scenarios in the spreadsheet:
    OffshoreWind = param.Pathways2Net0.evaluate('GALE!P'+str(param.Pathways2Net0RowInds[state.step_count]-1))
    OffshoreWind = np.clip(OffshoreWind + action[0], OffshoreWind, 150)
    BlueHydrogen = param.Pathways2Net0.evaluate('GALE!X'+str(param.Pathways2Net0RowInds[state.step_count]-1))
    BlueHydrogen = np.clip(BlueHydrogen + action[1], BlueHydrogen, 270)
    GreenHydrogen = param.Pathways2Net0.evaluate('GALE!Y'+str(param.Pathways2Net0RowInds[state.step_count]-1))
    GreenHydrogen = np.clip(GreenHydrogen + action[2], GreenHydrogen, 252.797394)
    
    # set these newly actioned deployment numbers into the corresponding cells in 'Gale' spreadsheet of the compiled object:
    # Note: the compiled object is essentially graphs with vertices/nodes and edges for cells and their relations (formulae)
    # if a cell contains raw values/numbers, its cell map/address will already exist; but if a cell originally contains 
    # formula (referencing to other cells) but not value, its cell map/address does not readily exist, 
    # so it has to be evaluated first such that the cell map/address corresponding to the node/vertex can be generated:
    # pycel's error message if not evaluate to "initialize" the cell map/address: 
    # AssertionError: Address "GALE!P36" not found in the cell map. Evaluate the address, or an address that references it, to place it in the cell map.
    param.Pathways2Net0.evaluate('GALE!P'+str(param.Pathways2Net0RowInds[state.step_count]))
    param.Pathways2Net0.evaluate('GALE!X'+str(param.Pathways2Net0RowInds[state.step_count]))
    param.Pathways2Net0.evaluate('GALE!Y'+str(param.Pathways2Net0RowInds[state.step_count]))
    # also, before resetting the current year's deployment values, the capex opex revenue and emissions of the current year
    # have to be evaluated to initialize the cell's map/address:
    capex_all = np.float32([param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'41')])
    opex_all = np.float32([param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'42')])
    revenue_all = np.float32([param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'43')])
    emissions = np.float32(param.Pathways2Net0.evaluate('CCUS!'+param.Pathways2Net0ColumnInds[state.step_count]+'68'))
    # again, before resetting the following 3 values, the above all reward components have to be evaluated first,
    # and after the following resetting, the reward components need to be evaluated again to calculate based on the newly
    # reset values:
    param.Pathways2Net0.set_value('GALE!P'+str(param.Pathways2Net0RowInds[state.step_count]), OffshoreWind)
    param.Pathways2Net0.set_value('GALE!X'+str(param.Pathways2Net0RowInds[state.step_count]), BlueHydrogen)
    param.Pathways2Net0.set_value('GALE!Y'+str(param.Pathways2Net0RowInds[state.step_count]), GreenHydrogen)
    # extract current state.step_count's capex, opex, revenue for all 3 techs, and the emission/carbon tax:
    capex_all = np.float32([param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'24'), 
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'28'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'32'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'36'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'41')])
    opex_all = np.float32([param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'25'), 
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'29'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'33'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'37'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'42')])
    revenue_all = np.float32([param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'26'), 
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'30'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'34'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'38'),
                          param.Pathways2Net0.evaluate('Outputs!'+param.Pathways2Net0ColumnInds[state.step_count]+'43')])
    # For current testing only, not involving the IEV economic model work book: 
    # one can add new rows in the original 'Outputs' spreadsheets of 'Pathways to Net Zero.xlsx' to calculate the sum,
    # and then directly evaluate the single cell to get the sum, which may be faster or slower than this current approach
    # (that is, running param.Pathways2Net0.evaluate() for 5 times and then np.sum(), compared to compiling a slighly complicated
    # workbook with a new row to compute the sum, and then running param.Pathways2Net0.evaluate() for 1 time)
    # If the IEV economic model work book is needed, then these values have to be evaluated one by one & input to the IEV model
    emissions = np.float32(param.Pathways2Net0.evaluate('CCUS!'+param.Pathways2Net0ColumnInds[state.step_count]+'68'))
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
    weightedRewardComponents[5] = state.EconoImpact
    # Update: based on the workshop on 20 Sep, the jobs number calculation we discussed was a simple one: 
    # 25% of total costs are spent on salaries and the average salary is GBP 50,000. 
    # Since the monetary unit of capex, opex, revenue, decom, etc. is millions of GBP in the compiled Excel workbook, 
    # the jobs number in each year should be: 0.25 * (capex + opex + decomm) / 0.05, where the decomm is a fixed constant 1050:
    weightedRewardComponents[4] = 0.25 * (weightedRewardComponents[0] + weightedRewardComponents[1] + 1050)/0.05
    state.jobs_increment = weightedRewardComponents[-2] - state.jobs
    state.jobs = weightedRewardComponents[-2]
    # reward = np.sum(weightedRewardComponents) # sum up the weighted reward components
    # reward = weightedRewardComponents # for testing/checking all components separately, using test_reference_environment.py
    # reward = weightedRewardComponents[-1] - weightedRewardComponents[-3] # proposed reward formula: Reward = Total economic impact - emissions
    reward = weightedRewardComponents[2] - np.sum(weightedRewardComponents[[0,1,3]]) - 1050 # new reward formula: - (capex + opex + decomm - revenue) - emissions, where oil & gas decomm is a fixed constant 1050/year for all scenarios
    return state, reward, weightedRewardComponents

def verify_constraints(state):
    verify = True
    # Constraint 1: no decrease in jobs in excess of 25,000 per year
    if state.step_count > 1: # Note: the index of jobs in reward_types is changed from 3 to 4: capex first, then opex, revenue, emissions, jobs, total economic impact
        if state.weightedRewardComponents_all[-1][4] - state.weightedRewardComponents_all[-2][4] < -25000:
            verify = False;
    # Constraint 2: no decrease in jobs in excess of 37,500 per two years
    if state.step_count > 2: # Previously in the original env.py, the ordering of reward_types is capex first, then opex, revenue, jobs, emissions
        if state.weightedRewardComponents_all[-1][4] - state.weightedRewardComponents_all[-3][4] < -37500:
            verify = False;
    # Constraint 3: amount of deployment possible in a single year should be less than the maximum single-year capex in any scenario
    # which is the total capex from Storm in 2050 = 26390
    if state.step_count > 0:
        if state.weightedRewardComponents_all[-1][0] > 26390:
            verify = False;
    return verify

def randomise(state, action):
    pass
    # uncomment to apply multiplicative noise to reward sensitivities
    #param.IEV_RewardSensitivities *= 1
    # uncomment to apply random delay to implementation of IEV years
    #action[param.scenarios:] = np.random.default_rng().integers(np.array(action[param.scenarios:]), endpoint = True)
    

#def update_prediction_array(prediction_array):
    #prediction_array = prediction_array + 0.1 * np.random.randn(1,len(prediction_array))[0]
    #return prediction_array


def plot_episode(state, fname):
    fig, ax = plt.subplots(2, 2)

    # cumulative total rewards
    plt.subplot(221)
    plt.plot(np.cumsum(state.rewards_all))
    plt.xlabel("time, avg reward: " + str(np.mean(state.rewards_all)))
    plt.ylabel("cumulative reward") 
    plt.tight_layout()
    # could be expanded to include individual components of the reward

    # generator levels
    plt.subplot(222)
    plt.plot(np.array(state.observations_all)[:,:4]) # first 4 elements of observations are step counts and 3 IEV years
    # plt.plot(np.array(state.observations_all)[:,-2:]) # last 2 elements of observations are jobs and increments in jobs
    plt.xlabel("time")
    plt.ylabel("observations")
    plt.tight_layout()


    # actions
    plt.subplot(223)
    plt.plot(np.array(state.actions_all))
    plt.xlabel("time")
    plt.ylabel("actions")
    plt.tight_layout()
    
    # jobs
    plt.subplot(224)
    plt.plot(np.vstack((np.array(state.weightedRewardComponents_all)[:,4],np.hstack((np.nan,np.diff(np.array(state.weightedRewardComponents_all)[:,4]))))).T)
    plt.xlabel("time")
    plt.ylabel("jobs and increments")
    plt.tight_layout()
    
    # # increments in jobs
    # plt.subplot(235)
    # plt.plot(np.diff(np.array(state.weightedRewardComponents_all)[:,4]))
    # plt.xlabel("time")
    # plt.ylabel("increments in jobs")
    # plt.tight_layout()

    # agent predictions
    #plt.subplot(224)
    #plt.plot(np.array(state.agent_predictions_all))
    #plt.xlabel("time")
    #plt.ylabel("predictions")
    #plt.tight_layout()


    plt.savefig(fname)

def score(state):
    value1 = np.sum(state.rewards_all)
    return {"value1" : value1}



class GymEnv(gym.Env):
    def __init__(self):
        self.seed()
        self.initialise_state()

    def initialise_state(self):
        self.state = State(seed=self.current_seed)
        self.action_space = action_space()
        self.observation_space = observation_space(self)
        self.param = param

    def reset(self):
        self.initialise_state()
        observation = self.state.to_observation()
        return observation

    def step(self, action):
        self.state.step_count += 1
        #self.state.prediction_array = update_prediction_array(
        #    self.state.prediction_array
        #)
        randomise(self.state, action)
        self.state, reward, weightedRewardComponents = apply_action(action, self.state)
        if verify_constraints(self.state) == False:
            reward = -1000
        #self.state.set_agent_prediction()
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
        #print('the score to be returned is: ',score(self.state))
        #return score(self.state)

    def plot(self, fname="episode.png"):
        plot_episode(self.state, fname)

    def render(self):
        pass

    def close(self):
        pass


