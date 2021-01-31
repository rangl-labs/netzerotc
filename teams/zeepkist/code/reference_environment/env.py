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
    imbalance_cost_factor_high = 50
    imbalance_cost_factor_low = 7
    ramp_1_max = 0.2
    ramp_2_max = 0.5
    ramp_1_min = -0.2
    ramp_2_min = -0.5
    generator_1_max = 3
    generator_1_min = 0.5
    generator_2_max = 2
    generator_2_min = 0.5
    generator_1_cost = 1
    generator_2_cost = 5

    # time parameters
    steps_per_episode = 96
    first_peak_time = 5


param = Parameters()  # parameters singleton


class State:
    def __init__(self, seed=None):
        np.random.seed(seed=seed)
        self.initialise_state()

    def reset(self):
        self.initialise_state()
        return self.to_observation()

    def initialise_state(self):
        # derived variables
        self.generator_1_level = 1.5
        self.generator_2_level = 0.5
        
        # forecast variables
        # initialise all predictions to 2
        self.prediction_array = np.full(param.steps_per_episode, 2)
        # sample the time of the second peak
        self.second_peak_time = np.random.randint(low=10, high=95)
        # add the peak demands
        self.prediction_array[param.first_peak_time] = 4
        self.prediction_array[self.second_peak_time] = 8
        
        self.agent_prediction = self.prediction_array

        # time variables
        # NOTE: our convention is to update step_count at the beginning of the gym step() function
        self.step_count = -1
        
        # histories
        self.observations_all = []
        self.actions_all = []
        self.rewards_all = []
        self.agent_predictions_all = []
        self.generator_1_levels_all = []
        self.generator_2_levels_all = []

    def to_observation(self):
        observation = (
            self.step_count,
            self.generator_1_level,
            self.generator_2_level,
        ) + tuple (self.agent_prediction)
        
        return observation

    def is_done(self):
        done = bool(
            self.step_count >= param.steps_per_episode - 1
            )
        return done

    def set_agent_prediction(self):
        self.agent_prediction = self.prediction_array
        


def record(state, action, reward):
    state.observations_all.append(state.to_observation())
    state.actions_all.append(action)
    state.rewards_all.append(reward)
    state.agent_predictions_all.append(state.agent_prediction)
    state.generator_1_levels_all.append(state.generator_1_level)
    state.generator_2_levels_all.append(state.generator_2_level)

def observation_space():
    obs_low = np.full(99, -1000, dtype=np.float32) # last 96 entries of observation are the predictions
    obs_low[0] = -1	# first entry of obervation is the timestep
    obs_low[1] = 0.5	# min level of generator 1 
    obs_low[2] = 0.5	# min level of generator 2
    obs_high = np.full(99, 1000, dtype=np.float32) # last 96 entries of observation are the predictions
    obs_high[0] = param.steps_per_episode	# first entry of obervation is the timestep
    obs_high[1] = 3	# max level of generator 1 
    obs_high[2] = 2	# max level of generator 2
    result = spaces.Box(obs_low, obs_high, dtype=np.float32)
    return result


def action_space():
    act_low = np.array(
        [
            0,
            0,
        ],
        dtype=np.float32,
    )
    act_high = np.array(
        [
            3,
            2,
        ],
        dtype=np.float32,
    )
    result = spaces.Box(act_low, act_high, dtype=np.float32)
    return result


def apply_action(action, state):

    # implement the generation levels requested by the agent 
    state.generator_1_level = action[0]
    state.generator_2_level = action[1]

    # check the previous generation levels
    if state.step_count == 0:
        state.generator_1_previous = 1.5
        state.generator_2_previous = 0.5
    else:    
        state.generator_1_previous = state.generator_1_levels_all[state.step_count - 1]
        state.generator_2_previous = state.generator_2_levels_all[state.step_count - 1]

    # calculate ramp rates 
    generator_1_ramp = state.generator_1_level - state.generator_1_previous
    generator_2_ramp = state.generator_2_level - state.generator_2_previous

    # curtail the actions if they exceed the ramp rate constraints
    if generator_1_ramp > param.ramp_1_max:
        state.generator_1_level = state.generator_1_previous + param.ramp_1_max
    if generator_1_ramp < param.ramp_1_min:
        state.generator_1_level = state.generator_1_previous + param.ramp_1_min
    if generator_2_ramp > param.ramp_2_max:
        state.generator_2_level = state.generator_2_previous + param.ramp_2_max
    if generator_2_ramp < param.ramp_2_min:
        state.generator_2_level = state.generator_2_previous + param.ramp_2_min

    # curtail the actions if they exceed the generator level constraints
    if state.generator_1_level > param.generator_1_max:
        state.generator_1_level = param.generator_1_max
    if state.generator_1_level < param.generator_1_min:
        state.generator_1_level = param.generator_1_min
    if state.generator_2_level > param.generator_2_max:
        state.generator_2_level = param.generator_2_max
    if state.generator_2_level < param.generator_2_min:
        state.generator_2_level = param.generator_2_min

    return state


def to_reward(state, done):
    # imbalance = total generation level - current demand level
    imbalance = state.generator_1_level + state.generator_2_level - state.prediction_array[state.step_count]
    if imbalance > 0:
        imbalance_cost = imbalance * param.imbalance_cost_factor_low
    else:
        imbalance_cost = -1 * imbalance * param.imbalance_cost_factor_high
    fuel_cost = param.generator_1_cost * state.generator_1_level + param.generator_2_cost * state.generator_2_level
    reward = 0 - imbalance_cost - fuel_cost
    
    # uncomment for Warmup phase:
    # if state.step_count != 1:
    #     reward = 0 
    
    # uncomment for Peak phase:
    # if state.step_count != 5:
    #     reward = 0
    
    return reward


def update_prediction_array(prediction_array):
    prediction_array = prediction_array + 0.1 * np.random.randn(1,len(prediction_array))[0]
    return prediction_array


def plot_episode(state, fname):
    fig, ax = plt.subplots(2, 2)

    # cumulative total cost
    plt.subplot(221)
    plt.plot(np.cumsum(state.rewards_all))
    plt.xlabel("time")
    plt.ylabel("cumulative reward")
    plt.tight_layout()
    # could be expanded to include individual components of the reward

    # generator levels
    plt.subplot(222)
    plt.plot(np.array(state.generator_1_levels_all))
    plt.plot(np.array(state.generator_2_levels_all))
    plt.xlabel("time")
    plt.ylabel("generator levels")
    plt.tight_layout()


    # actions
    plt.subplot(223)
    plt.plot(np.array(state.actions_all))
    plt.xlabel("time")
    plt.ylabel("actions")
    plt.tight_layout()


    # agent predictions
    plt.subplot(224)
    plt.plot(np.array(state.agent_predictions_all))
    plt.xlabel("time")
    plt.ylabel("predictions")
    plt.tight_layout()


    plt.savefig(fname)

def score(state):
    value1 = np.sum(state.rewards_all)
    return {"value1" : value1}


# TODO
# def fetch_prediction(prediction_array, step):
#     prediction = prediction_array[step]
#     return prediction


class GymEnv(gym.Env):
    def __init__(self):
        self.seed()
        self.initialise_state()

    def initialise_state(self):
        self.state = State(seed=self.current_seed)
        self.action_space = action_space()
        self.observation_space = observation_space()
        self.param = param

    def reset(self):
        self.initialise_state()
        observation = self.state.to_observation()
        return observation

    def step(self, action):
        self.state.step_count += 1
        self.state.prediction_array = update_prediction_array(
            self.state.prediction_array
        )
        self.state = apply_action(action, self.state)
        self.state.set_agent_prediction()
        observation = self.state.to_observation()
        done = self.state.is_done()
        reward = to_reward(self.state, done)
        record(self.state, action, reward)
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

