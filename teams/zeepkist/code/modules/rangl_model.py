# -*- coding: utf-8 -*-
"""
Skeleton rangl environment, adapted from reference_environment/env.py. 
Used for submission only (when the reference_environment is not available). 

Target: python 3.8
@author: Simon Tindemans
Delft University of Technology
s.h.tindemans@tudelft.nl
"""
# SPDX-License-Identifier: MIT

class Parameters:
    """Singleton class that is used to store problem parameters"""
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

class SkeletonEnvironment:
    """
    Class that mirrors the essential elements of the reference gym environment. 
    
    It includes the self.forecast_length variable set by the ObservationWrapper.
    """

    param = Parameters()  # parameters singleton

    # set the forecast_length variable
    def __init__(self, forecast_length=None):
        if forecast_length is None:
            self.forecast_length = self.param.steps_per_episode
        else:
            self.forecast_length = forecast_length
        return



