#%% Imports

import time
import random
import csv

import pandas as pd
import numpy as np
import gym
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

#%% Classes

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, params):
        super().__init__(env)
        self.look_ahead = params['look_ahead']
        self.shift = params['shift']
        self.finish = params['finish']
        self.v = params['v']
    
    def observation(self, obs):
        copy = np.array(obs, dtype=np.float32)
        self.v = self.v - 1 if copy[0] > (self.finish - (self.shift + 1) - self.look_ahead) else self.v
        copy[self.shift : obs[0] + self.shift + 1] = 0
        # copy[obs[0] + self.shift + 1 + self.v : self.finish] = 0
        return tuple(copy)

class Trainer:
    def __init__(self, env, params):
        self.model = SAC(MlpPolicy, env, verbose=1, **params)
        self.param = env.param

    def train_rl(self, models_to_train=1, episodes_per_model=100):
        start = time.time()

        for i in range(models_to_train):
            steps_per_model = episodes_per_model * self.param.steps_per_episode
            self.model.learn(total_timesteps=steps_per_model)
            self.model.save("MODEL_" + str(i))

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

class EvaluateWithConverter:
    def __init__(self, env, agent=None, converter=None):
        self.env = env
        self.param = env.param
        self.agent = agent
        self.converter = converter

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

    def RL_agent(self, seeds):
        rewards = []
        model = self.agent
        for seed in seeds:
            self.env.seed(seed)
            obs = self.env.reset()
            while not self.env.state.is_done():
                action, _states = model.predict(self.converter.transform_obs(obs), deterministic=True)
                obs, _, _, _ = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)