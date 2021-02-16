import numpy as np
import csv
import torch
import gym
import numpy as np
from ddpg_agent_training.utils.models import actor

def read_seeds(fname="seeds.csv"):
    file = open(fname)
    csv_file = csv.reader(file)
    seeds = []
    for row in csv_file:
        seeds.append(int(row[0]))
    return seeds

def _preproc_inputs(obs):
    inputs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return inputs


if __name__ == '__main__':
    # Create an environment
    env = gym.make("reference_environment:reference-environment-v0")
    seeds = read_seeds()
    # environment parameters
    env_params = {'obs': env.observation_space.shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high,
                  'reward': 1,
                  'max_timesteps': 96
                  }

    n_test_rollouts = 1

    model = torch.load("trained_models/model.pt")

    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()

    total_reward = []
    for seed in seeds:
        env.seed(seed)
        for _ in range(n_test_rollouts):
            per_reward = []
            observation = env.reset()
            obs = observation
            while not env.state.is_done():
                # for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = _preproc_inputs(obs)
                    pi = actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, done, _ = env.step(actions)
                obs = observation_new
            total_reward.append(sum(env.state.rewards_all))

    print(np.mean(total_reward))
