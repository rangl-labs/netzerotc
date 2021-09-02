import math

import pandas as pd
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
import matplotlib.pyplot as plt


class Parameters:
    # (Avoid sampling random variables here: they would not be resampled upon reset())
    # problem-specific parameters
    
    techs = 2 # number of technologies
    scenarios = 3 # number of strategies ('scenarios' in the IEV terminology, eg Breeze, Gale, Storm)
    reward_types = 6 # capex first, then opex, revenue, jobs, emissions, total economic impact
    steps_per_episode = 30 # number of years in the planning horizon (eg. 2021 -> 2050 = 30)
    IEV_Rewards = np.ones((scenarios, steps_per_episode, reward_types)) # rewards for each scenario in each year in the IEV model, by reward type (capex first, then opex, revenue, ...)
    IEV_RewardSensitivities = np.full_like(IEV_Rewards,1.01) # in any strategy, when any investment is brought forward by one year, its associated rewards must be multiplied by this factor to account for changing costs/reevenues over time
    
    IEV_Rewards[:,:,0] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total capex.xlsx'))[:,4:],dtype=np.float64) # [:,4:] corresponds to 2021 -> 2050, 30 values/steps in total
    IEV_Rewards[:,:,1] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total opex.xlsx'))[:,4:],dtype=np.float64)
    IEV_Rewards[:,:,2] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Total revenue.xlsx'))[:,4:],dtype=np.float64)
    IEV_Rewards[:,:,3] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Jobs.xlsx'))[:,2:],dtype=np.float64) # for jobs only, [:,2:] corresponds to 2021 -> 2050, 30 values/steps in total
    IEV_Rewards[:,:,4] = np.array(np.array(pd.read_excel('./sensitivities/Pathways to Net Zero - Original - Carbon tax for uncaptured carbon.xlsx'))[:,4:],dtype=np.float64)
    IEV_Rewards[:,:,5] = np.array(np.array(pd.read_excel('./sensitivities/IEV - Original - Total Economic Impact.xlsx'))[:,4:],dtype=np.float64)
    IEV_RewardSensitivities[:,0:-1,0] = np.genfromtxt('./sensitivities/Pathways to Net Zero - Total capex Sensitivity Ratio for 1-Year Shifting.csv',delimiter=',') # 0:-1 corresponds to 2021 -> 2049
    IEV_RewardSensitivities[:,0:-1,1] = np.genfromtxt('./sensitivities/Pathways to Net Zero - Total opex Sensitivity Ratio for 1-Year Shifting.csv',delimiter=',')
    IEV_RewardSensitivities[:,0:-1,2] = np.genfromtxt('./sensitivities/Pathways to Net Zero - Total revenue Sensitivity Ratio for 1-Year Shifting.csv',delimiter=',')
    IEV_RewardSensitivities[:,0:-1,3] = np.genfromtxt('./sensitivities/IEV - Total Jobs Sensitivity Ratio for 1-Year Shifting.csv',delimiter=',') # including direct and indirect, but without induced; may read the "IEV - Total Jobs Including Induced Sensitivity Ratio for 1-Year Shifting.csv" to load the sensitivity for induced jobs included.
    IEV_RewardSensitivities[:,0:-1,4] = np.genfromtxt('./sensitivities/Pathways to Net Zero - Total emissions (uncaptured carbon tax) Sensitivity Ratio for 1-Year Shifting.csv',delimiter=',')
    IEV_RewardSensitivities[:,0:-1,5] = np.genfromtxt('./sensitivities/IEV - Total Economic Impact Sensitivity Ratio for 1-Year Shifting.csv',delimiter=',')


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
        self.IEV_years = np.zeros(param.scenarios, dtype=int) # for each scenario, records the latest IEV year that has been implemented
        
        # histories
        self.observations_all = []
        self.actions_all = []
        self.rewards_all = []
        self.weightedRewardComponents_all = [] 

    def to_observation(self):
        observation = (
            self.step_count,
        ) + tuple (self.IEV_years)
        
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
    obs_high = np.full_like(self.state.to_observation(), 1000, dtype=np.float32) 
    obs_high[0] = param.steps_per_episode	# first entry of obervation is the timestep
    result = spaces.Box(obs_low, obs_high, dtype=np.float32)
    return result


def action_space():
    act_low = np.zeros(2 * param.scenarios, dtype=np.float32) # scenario weights and scenario years to advance
    act_high = np.full(2 * param.scenarios, param.steps_per_episode, dtype=np.float32)
    result = spaces.Box(act_low, act_high, dtype=np.float32)
    return result


def apply_action(action, state):

    # calculate the rewards accruing to each scenario
    scenarioWeights = np.zeros(param.scenarios)
    if state.step_count == 0:
        scenarioWeights = action[:param.scenarios]
    scenarioYears = action[param.scenarios:]
    # prevent advancing beyond end of scenario
    for scenario in np.arange(param.scenarios):
        if state.IEV_years[scenario] + scenarioYears[scenario] >= param.steps_per_episode:
            scenarioYears[scenario] = param.steps_per_episode - 1 - state.IEV_years[scenario]
    #capex = 0 # this variable will aggregate all (rebased) capital expenditure for this time step
    rewardComponents = np.zeros((param.scenarios, param.reward_types)) # for each scenario, this array will hold all components of reward for this time step
    IEV_LastRewards = 0 # this variable will aggregate all other rewards for this time step (these rewards are all assumed to be annual rates)
    
    for scenario in np.arange(param.scenarios): # for each scenario
        for IEV_year in np.arange(state.IEV_years[scenario], state.IEV_years[scenario] + scenarioYears[scenario]): # for each IEV year to be implemented this time
            IEV_year = int(IEV_year)
            IEV_YearReward = param.IEV_Rewards[scenario,IEV_year,0] # get the raw capex
            for sensitivityYear in np.arange(state.step_count, IEV_year): 
                IEV_YearReward *= param.IEV_RewardSensitivities[scenario, sensitivityYear, 0] # apply each relevant sensitivity
            rewardComponents[scenario, 0] += IEV_YearReward
        # now deal with remaining rewards: include the annual rates from *only* the last IEV year to be implemented this time
        IEV_year = state.IEV_years[scenario] + scenarioYears[scenario] # identify the last IEV year to be implemented this time
        for rewardType in np.arange(1, param.reward_types): # for each remaining reward type (these should all represent annual rates rather than one-off charges/rewards, and by convention we apply the last rate)
            IEV_year = int(IEV_year)
            IEV_YearRate = param.IEV_Rewards[scenario,IEV_year,rewardType] # get the raw reward rate
            for sensitivityYear in np.arange(state.step_count, IEV_year): 
                IEV_YearRate *= param.IEV_RewardSensitivities[scenario, sensitivityYear, rewardType] # apply each relevant sensitivity
            rewardComponents[scenario, rewardType] = IEV_YearRate
    weightedRewardComponents = np.matmul(scenarioWeights, rewardComponents) # weight the reward components by the scenario weights
    state.IEV_years = np.clip(state.IEV_years + scenarioYears, 0, param.steps_per_episode - 1) # record the latest IEV year implemented
    reward = np.sum(weightedRewardComponents) # sum up the weighted reward components
    return state, reward, weightedRewardComponents

def verify_constraints(state):
    verify = True
    # Constraint 1: no decrease in jobs in excess of 25,000 per year
    if state.step_count > 1:
        if state.weightedRewardComponents_all[-1][3] - state.weightedRewardComponents_all[-2][3] < -25000:
            verify = False;
    # Constraint 2: no decrease in jobs in excess of 37,500 per two years
    if state.step_count > 2:
        if state.weightedRewardComponents_all[-1][3] - state.weightedRewardComponents_all[-3][3] < -37500:
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
    plt.plot(np.array(state.observations_all))
    plt.xlabel("time")
    plt.ylabel("observations")
    plt.tight_layout()


    # actions
    plt.subplot(223)
    plt.plot(np.array(state.actions_all))
    plt.xlabel("time")
    plt.ylabel("actions")
    plt.tight_layout()


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

