import time
import random
import csv

import pandas as pd
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
# from pathlib import Path


class Trainer:
    def __init__(self, env):
        self.env = env
        self.param = env.param

    def train_rl(self, models_to_train=40, episodes_per_model=100, last_model_number=-1):
        # specify the RL algorithm to train (eg ACKTR, TRPO...)
        model = PPO(MlpPolicy, self.env, verbose=1)
        if last_model_number > -1:
            # last_model_number = 39
            model.load("MODEL_" + str(last_model_number))
        start = time.time()

        for i in range(last_model_number + 1, last_model_number + 1 + models_to_train):
            steps_per_model = episodes_per_model * self.param.steps_per_episode
            model.learn(total_timesteps=steps_per_model)
            model.save("MODEL_" + str(i))

        end = time.time()
        print("time (min): ", (end - start) / 60)


class Evaluate:
    def __init__(self, env, agent=None):
        self.env = env
        self.param = env.param
        self.agent = agent

    def generate_random_seeds(self, n, fname="test_set_seeds.csv"):
        seeds = [random.randint(0, 1e7) for i in range(n)]
        df = pd.DataFrame(seeds)
        df.to_csv(fname, index=False, header=False)

    def read_seeds(self, fname="test_set_seeds.csv"):
        file = open(fname)
        csv_file = csv.reader(file)
        seeds = []
        for row in csv_file:
            seeds.append(int(row[0]))
        self.seeds = seeds
        return seeds

    def min_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                if type(self.env.action_space) == gym.spaces.discrete.Discrete:
                    action = 0
                elif type(self.env.action_space) == gym.spaces.Box:
                    action = self.env.action_space.low
                    # spaces gym.spaces.MultiDiscrete, gym.spaces.Tuple not yet covered
                # spaces gym.spaces.MultiDiscrete, gym.spaces.Tuple not yet covered
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def max_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                if type(self.env.action_space) == gym.spaces.discrete.Discrete:
                    action = self.env.action_space.n - 1
                elif type(self.env.action_space) == gym.spaces.Box:
                    action = self.env.action_space.high
                # spaces gym.spaces.MultiDiscrete, gym.spaces.Tuple not yet covered
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def breeze_agent(self, seeds):
        rewards = []
        # deployments = np.array(np.array(pd.read_excel('BREEZE_Deployments.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
        deployments = np.array(np.array(pd.read_excel('BREEZE_Deployments_Modified.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
        actions = deployments[1:,:] - deployments[:-1,:]
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                action = actions[self.env.state.step_count + 1]
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def gale_agent(self, seeds):
        rewards = []
        # deployments = np.array(np.array(pd.read_excel('GALE_Deployments.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
        deployments = np.array(np.array(pd.read_excel('GALE_Deployments_Modified.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
        actions = deployments[1:,:] - deployments[:-1,:]
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                action = actions[self.env.state.step_count + 1]
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def storm_agent(self, seeds):
        rewards = []
        # deployments = np.array(np.array(pd.read_excel('STORM_Deployments.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
        deployments = np.array(np.array(pd.read_excel('STORM_Deployments_Modified.xlsx'))[-(self.env.param.steps_per_episode+1):,1:],dtype=np.float32)
        actions = deployments[1:,:] - deployments[:-1,:]
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                action = actions[self.env.state.step_count + 1]
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def random_agent(self, seeds):
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            while not self.env.state.is_done():
                ###
                action = self.env.action_space.sample()
                ###
                self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        # TODO we should double check that sampling from the observation space is independent from
        # sampling in the environment which happens with fixed seed
        return np.mean(rewards)

    def RL_agent(self, seeds):
        rewards = []
        model = self.agent
        for seed in seeds:
            self.env.seed(seed)
            obs = self.env.reset()
            while not self.env.state.is_done():
                action, _states = model.predict(obs, deterministic=True)
                obs, _, _, _ = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)
