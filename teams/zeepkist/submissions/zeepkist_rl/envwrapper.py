# -*- coding: utf-8 -*-
"""
Gym environment wrapper functions for observations and actions

Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

import gym
import numpy as np

def obs_transform(observation, env_model):
    # explicit mapping to list, to deal with local/online platform differences
    obs_list = list(observation)
    obs_header = obs_list[0:3]
    # get the current time (starting from minus one) and map it to an integer
    current_time = int(round(obs_header[0]))
    # take only the part that is an actual forecast (larger than t)
    # repeat the final element so that we do not run out of observations when t=95 (final invocation after the last action)
    obs_forecast = obs_list[3 + current_time + 1 : ] + [observation[-1]]
    # pad the forecast with copies of the final value (create an array of size 97)
    padding_required = (len(obs_list) - 3) + 1 - len(obs_forecast)
    padded_forecast = np.pad(obs_forecast, (0,padding_required), 'edge')
    # return an observation vector of the same length
    return obs_header + list(padded_forecast)[:env_model.forecast_length]

def act_transform(action, env_model, current_gen_level):
    offset_1 = action[0]*env_model.param.ramp_1_max if action[0] >= 0 else - action[0]*env_model.param.ramp_1_min
    offset_2 = action[1]*env_model.param.ramp_2_max if action[1] >= 0 else - action[1]*env_model.param.ramp_2_min
    a1 = current_gen_level[0] + offset_1
    a2 = current_gen_level[1] + offset_2
    return (a1, a2)


class EfficientObsWrapper(gym.ObservationWrapper):
    """
    Wrapper for observations.

    Modifies the observation vector so that the forecast part (values 3:forecast_length+3) consists of the current value and 
    immediate forecast, padded with repeats of the final element if necessary. This should make the policy time-invariant.
    """

    def __init__(self, env, forecast_length=25):
        super().__init__(env)
        assert 1 <= forecast_length <= self.env.param.steps_per_episode,\
            f"Observation length {forecast_length} is outside the permissible range (1, {self.env.param.steps_per_episode})"
        self.forecast_length = forecast_length
        self.observation_space = self._obs_space()

    def observation(self, observation):
        return obs_transform(observation, env_model=self)

    def _obs_space(self):
        # modified from the base environment to restrict the observation length
        obs_low = np.full(self.forecast_length + 3, -1000, dtype=np.float32) # last 'forecast_length' entries of observation are the predictions
        obs_low[0] = -1	# first entry of obervation is the timestep
        obs_low[1] = self.env.param.generator_1_min	# min level of generator 1 
        obs_low[2] = self.env.param.generator_2_min	# min level of generator 2
        obs_high = np.full(self.forecast_length + 3, 1000, dtype=np.float32) # last 96 entries of observation are the predictions
        obs_high[0] = self.env.param.steps_per_episode	# first entry of obervation is the timestep
        obs_low[1] = self.env.param.generator_1_max	# max level of generator 1 
        obs_low[2] = self.env.param.generator_2_max	# max level of generator 2
        result = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)
        return result


# class ObsWrapper(gym.ObservationWrapper):
#     """
#     Wrapper for observations.

#     Modifies the observation vector so that the forecast part (values 2:98) consists of the current value and 
#     immediate forecast, padded with repeats of the final element. This should make the policy time-invariant.
#     """

#     def __init__(self, env):
#         super().__init__(env)

#     def observation(self, observation):
#         obs_header = observation[0:3]
#         # include the 'current' value for t=-1 for consistency
#         extended_demand_series = (2,) + observation[3:]
#         # take only the part from t until the future
#         obs_forecast = extended_demand_series[obs_header[0] + 1:]
#         # pad the forecast with copies of the final value
#         padding_required = self.env.param.steps_per_episode + 1 - len(obs_forecast)
#         padded_forecast = np.pad(obs_forecast, (0,padding_required), 'edge')
#         # return an observation vector of the same length
#         return obs_header + tuple(padded_forecast)[:self.env.param.steps_per_episode]


class ActWrapper(gym.ActionWrapper):
    """
    Wrapper for actions.

    Adjust the actions to lie in a [-1,1] 2D box, adapted to the ramp rates of the generators.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1,1,(2,))

    def action(self, action):
        return act_transform(action, self.env, (self.env.state.generator_1_level, self.env.state.generator_2_level))
