#!/usr/bin/env python

import os
import gym
import csv
import argparse
import math
import numpy as np
import copy
import datetime

# implementation of sequential golden section search
def sequential_golden_section_search(
    env,
    search_iterations,
    search_direction,
    search_seeds,
    search_params,
    search_results_folder,
    search_results_rounding):
    
    num_time_steps = 20
    num_actions = 3
    the_golden_ratio = (math.sqrt(5) + 1) / 2

    # initialize the parameters array with the lowest values
    parameters = np.zeros((num_time_steps, num_actions))
    for time_step in range(num_time_steps):
        parameters[time_step] = env.action_space.low

    # evaluate the mean return of a profile with the given seeds
    def get_mean_returns(seeds, strategy_parameters):
        rewards = []
        for seed in seeds:
            env.seed(seed)
            env.reset()
            step_count = 0
            while not env.state.is_done():
                sampled_action = strategy_parameters[step_count]
                env.step(sampled_action)
                step_count += 1
            rewards.append(sum(env.state.rewards_all))
        return np.mean(rewards)

    # perform golden section search on the action of index `action_index``, at time step `step``
    def golden_section_search(step, action_index):
        """
        Adapted from https://en.wikipedia.org/wiki/Golden-section_search
        """

        def eval(seeds, value):
            strategy_parameters = copy.deepcopy(parameters)
            strategy_parameters[step][action_index] = value
            return get_mean_returns(seeds, strategy_parameters)

        tol = search_params["tol"]
        max_num_steps = search_params["max_num_steps"]
        num_seeds_per_eval = search_params["num_seeds_per_eval"]
        if num_seeds_per_eval == -1:
            num_seeds_per_eval = len(search_seeds)
        
        # decide the starting point of the search
        min_num = env.action_space.low[action_index]
        max_num = env.action_space.high[action_index]

        low = min_num
        high = max_num

        near_low = high - (high - low) / the_golden_ratio
        near_high = low + (high - low) / the_golden_ratio
        
        step_so_far = 0
        while abs(near_high - near_low) > tol and step_so_far < max_num_steps:
            print("search in the range", low, high, flush=True)
            seeds = search_seeds[:num_seeds_per_eval]
            near_low_return = eval(seeds, near_low)
            near_high_return = eval(seeds, near_high)
            print("step {}".format(step_so_far))
            print("evaluation", near_low, near_low_return)
            print("evaluation", near_high, near_high_return)
            if near_low_return > near_high_return:
                high = near_high
            else:
                low = near_low

            near_low = high - (high - low) / the_golden_ratio
            near_high = low + (high - low) / the_golden_ratio

            step_so_far += 1
        
        final_value = (high+low)/2
        return final_value

    # perform the golden section search sequentially for a number of iterations
    for i_iter in range(search_iterations):
        step_generator = range(0, num_time_steps, 1)
        print("iteration {} started.".format(i_iter))
        if search_direction == "backward":
            step_generator = reversed(step_generator)
        for i_step in step_generator:
            for i_action in range(num_actions):
                now = datetime.datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("{} searching".format(current_time), i_step, i_action, flush=True)
                parameters[i_step][i_action] = golden_section_search(i_step, i_action)
                print("selected: {}".format(parameters[i_step][i_action]))
                # update the parameters
                now = datetime.datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("{} return of the current profile: {}".format(current_time, get_mean_returns(search_seeds, parameters)))
                print('--------------------------------------------------------')

        # save the parameters
        save_path = "{}/search_result_iter_{}.txt".format(search_results_folder, i_iter)
        with open(save_path, "w") as f:
            for i_step in range(num_time_steps):
                for i_action in range(num_actions):
                    f.write("{} ".format(parameters[i_step][i_action]))
                f.write("\n")
            print("parameters saved at {}".format(save_path), flush=True)

    # save the final parameters with rounding
    save_path = "{}/search_result_final.txt".format(search_results_folder)
    with open(save_path, "w") as f:
        for i_step in range(num_time_steps):
            for i_action in range(num_actions):
                f.write("{} ".format(round(parameters[i_step][i_action], search_results_rounding)))
            f.write("\n")
    print("Sequential golden section search was executed successfully. The final result is stored at {}".format(save_path))

# the argument parser
parser = argparse.ArgumentParser()

# evaluation seeds
parser.add_argument('--seeds_fname', type=str, default="seeds.csv", help="name of the file that stores the seeds for the evaluation of strategies")

# path to store results
parser.add_argument('--results_folder', type=str, default="saved_models", help="the path to the folder where we store the results")

# options of sequential golden section search
parser.add_argument('--search_direction', type=str, choices=["forward", "backward"], default="backward", help="the direction of sequential golden section search. it decides if the search will be conducted from the first time step (forward) or the last time step(backward)")
parser.add_argument('--search_iterations', type=int, default=1, help="the number of search iterations. in each iteration, we optimize through the entire parameter space. our experiments reveal that one iteration is generally sufficient.")

# hyperparameters of golden section search
parser.add_argument('--num_seeds_per_eval', type=int, default=-1, help="how many seeds do we use for the evaluation of each point")
parser.add_argument('--tol', type=float, default=0.01, help="the tolerance level of golden section search")
parser.add_argument('--max_num_steps', type=int, default=100, help="the maximum number of steps in one golden section search")

parser.add_argument('--rounding', type=int, default=2, help="how many digits we round the final results to?")

# parse the arguments
args = parser.parse_args()

# read environment seeds for the evaluation of strategies
seeds_fname = args.seeds_fname
def read_seeds(fname):
    file = open(fname)
    csv_file = csv.reader(file)
    seeds = []
    for row in csv_file:
        seeds.append(int(row[0]))
    return seeds
seeds = read_seeds(fname=seeds_fname)
print("evaluation seeds:", seeds)

# create result folder if not already existing
results_folder = args.results_folder
if os.path.exists(results_folder) is False:
    os.makedirs(results_folder, exist_ok=False)

# initialize the env - need it for evaluation
env = gym.make("rangl:nztc-open-loop-v0")

# execute the sequential golden section search
search_params = {
    "tol": args.tol,
    "max_num_steps": args.max_num_steps,
    "num_seeds_per_eval": args.num_seeds_per_eval
}
sequential_golden_section_search(
    env=env, 
    search_iterations=args.search_iterations, 
    search_direction=args.search_direction, 
    search_seeds=seeds, 
    search_params=search_params,
    search_results_folder=args.results_folder,
    search_results_rounding=args.rounding)

