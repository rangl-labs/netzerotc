import time
import random
import csv
import pandas as pd
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from gym import spaces, ObservationWrapper, RewardWrapper, ActionWrapper
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Trainer:
    def __init__(self, env):
        self.env = env
        self.param = env.param

    def train_rl(self, models_to_train=40, episodes_per_model=100, path='./logs/'):
        # specify the RL algorithm to train (eg ACKTR, TRPO...)

        # Callback for saving the best agent during training
        eval_callback = EvalCallback(self.env, best_model_save_path=path,
                                     log_path=path, eval_freq=500,
                                     deterministic=True, render=False)

        model = PPO(MlpPolicy, self.env, verbose=1, learning_rate=0.0003, tensorboard_log=path)
        start = time.time()

        for i in range(models_to_train):
            steps_per_model = episodes_per_model * self.param.steps_per_episode
            model.learn(total_timesteps=steps_per_model, callback=eval_callback)
            model.save("MODEL_" + str(i))

        end = time.time()
        print("time (min): ", (end - start) / 60)

    def retrain_rl(self, model, episodes, path='./logs/'):
        # Method for retraining a saved model for more timesteps
        start = time.time()

        # Callback for saving the best agent during training
        eval_callback = EvalCallback(self.env, best_model_save_path=path,
                                     log_path=path, eval_freq=500,
                                     deterministic=True, render=False)
        steps_per_model = episodes * self.param.steps_per_episode

        model.set_env(self.env)
        model.learn(total_timesteps=steps_per_model, callback=eval_callback)
        model.save("MODEL_RETRAINED")

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
                action, _states = model.predict(obs,deterministic=True)
                obs, _, _, _ = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def matching_agent(self, seeds):
        # This uses the expensive generator to poorly match the predicted generation.
        rewards = []
        for seed in seeds:
            self.env.seed(seed)
            self.env.reset()
            obs = self.env.step([2,0])
            while not self.env.state.is_done():
                current_time = obs[0][0]
                current_generation_1 = obs[0][1]
                current_generation_2 = obs[0][2]
                forecasts = obs[0][3:]
                predicted_generation = forecasts[current_time]
                extra_generation = predicted_generation - current_generation_1 - current_generation_2
                action = [current_generation_1, current_generation_2 + extra_generation]
                obs = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

    def transformed_agent(self, seeds, H, transform):
        rewards = []
        model = self.agent
        for seed in seeds:
            self.env.seed(seed)
            obs = self.env.reset()
            while not self.env.state.is_done():
                obs = ObservationTransform(obs, H, transform)
                action, _states = model.predict(obs, deterministic=True)
                obs, _, _, _ = self.env.step(action)
            rewards.append(sum(self.env.state.rewards_all))
        return np.mean(rewards)

### The reason I am defining this ObservationMapping as a separate function and not a method is
### because we want the exact same function to be used when running the evaluation of the agent.
### Particularly important when submitting the agent to Rangl.

def ObservationTransform(obs, H, transform, steps_per_episode=int(96)):
    step_count, generator_1_level, generator_2_level = obs[:3]
    agent_prediction = np.array(obs[3:])  # since it was stored as a tuple

    agent_horizon_prediction = agent_prediction[-1] * np.ones(steps_per_episode)
    agent_horizon_prediction[:int(steps_per_episode - step_count)] = agent_prediction[int(step_count):]  # inclusive index
    agent_horizon_prediction = agent_horizon_prediction[:H]

    if transform == "Standard":
        pass
    if transform == "Zeroed":
        agent_horizon_prediction -= agent_prediction[step_count] * np.ones(H)
    if transform == "Deltas":
        # TODO: test this
        agent_horizon_prediction = np.concatenate(([agent_prediction[step_count]],
                                                   agent_horizon_prediction))
        agent_horizon_prediction = np.diff(agent_horizon_prediction)

    obs = (step_count, generator_1_level, generator_2_level) + tuple(agent_horizon_prediction)

    return obs


class HorizonObservationWrapper(ObservationWrapper):

    def __init__(self, env, horizon_length, transform_name):

        super(HorizonObservationWrapper, self).__init__(env)

        self.H = horizon_length

        # Different transform methods
        transform_options = ["Standard", "Zeroed", "Deltas"]
        assert transform_name in transform_options, "Set a valid transform"
        self.transform = transform_name

        self.steps_per_episode = int(96)
        self.n_obs = len(ObservationTransform( tuple(np.ones(99,)), self.H , transform=self.transform))
        self.observation_space = self.get_observation_space()

    def get_observation_space(self):

        obs_low = np.full(self.n_obs, -1000, dtype=np.float32)  # last 96 entries of observation are the predictions
        obs_low[0] = -1  # first entry of obervation is the timestep
        obs_low[1] = 0.5  # min level of generator 1
        obs_low[2] = 0.5  # min level of generator 2
        obs_high = np.full(self.n_obs, 1000, dtype=np.float32)  # last 96 entries of observation are the predictions
        obs_high[0] = self.param.steps_per_episode  # first entry of obervation is the timestep
        obs_high[1] = 3  # max level of generator 1
        obs_high[2] = 2  # max level of generator 2
        result = spaces.Box(obs_low, obs_high, dtype=np.float32)
        return result

    def observation(self, obs):

        # Apply the globally defined ObservationTransform transform to the observations
        obs = ObservationTransform(obs, self.H, transform=self.transform, steps_per_episode=self.steps_per_episode)

        return obs


class PhaseRewardWrapper(RewardWrapper):
    def __init__(self, env, phase="Full"):
        super(PhaseRewardWrapper, self).__init__(env)

        assert phase in ["Warmup", "Peak", "Full"], "Set valid phase."
        self.phase = phase

    def reward(self, rew):

        if self.phase=="Warmup" and self.env.state.step_count != 1:
            rew = 0

        if self.phase=="Peak" and self.env.state.step_count != 5:
            rew = 0

        return rew


class RandomActionWrapper(gym.ActionWrapper):

   def __init__(self, env, epsilon=0.1):

       super(RandomActionWrapper, self).__init__(env)

       self.epsilon = epsilon

   def action(self, action):
       if random.random() < self.epsilon:
           return self.env.action_space.sample()
       return action


class OurActionWrapper(ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

        act_low = np.array(
            [
                0.5,
                0.5,
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
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    def action(self, act):
        return act


class JoesActionWrapper(gym.ActionWrapper):

    def __init__(self, env):

        super(JoesActionWrapper, self).__init__(env)
        act_low = np.array(
            [
                -0.7
            ],
            dtype=np.float32,
        )
        act_high = np.array(
            [
                0.7
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

    def action(self, action):
        """
        Takes the desired change in total power generated by the system and outputs a tuple containing
        the 'optimal' action for both generators.
        delta \in [-0.7, 0.7] will saturate if value given outside this range.
        """
        delta = action
        if delta < -0.7:
            return (-0.2, -0.5)
        elif delta >= -0.7 and delta < -0.3:
            return (delta + 0.5, -0.5)
        elif delta >= -0.3 and delta <= 0.7:
            return (0.2, delta - 0.2)
        # else delta > 0.7
        return (0.2, 0.5)


def get_agent_prediction(agent_predictions):
    """
    :arg: agent_predictions (timesteps, timesteps) array like
    :returns: agent_prediction (timesteps, timesteps) array like
    """
    steps = np.shape(agent_predictions)[0]
    outruns = []
    plot_mes = []
    for t in range(steps):
        plot_me = np.empty(steps)
        # plot the
        plot_me[t:] = agent_predictions[t, t:]
        plot_me[:t] = np.array(outruns)
        outruns.append(agent_predictions.T[t, t])
        plot_mes.append(plot_me)

    return np.array(plot_mes)


def plot_frame(state_tuple, lim_tuple, ax, frame):
    (rewards_total,
     # rewards_fuel_cost,
     # rewards_imbalance_cost,
     generator_1_levels,
     generator_2_levels,
     actions,
     agent_predictions,
     agent_prediction) = state_tuple
    (xlim_max, ylim_min_1, ylim_max_3,
     ylim_min_3, ylim_max_4, ylim_min_4) = lim_tuple
    # Unpack ax tuple
    ((ax1, ax2), (ax3, ax4)) = ax

    # cumulative total cost
    ax1.set_xlim(0, xlim_max)
    ax1.set_ylim(ylim_min_1, 0)
    ax1.plot(np.cumsum(rewards_total[:frame + 1]))
    ax1.set_xlabel("time")
    ax1.set_ylabel("cumulative reward")
    # could be expanded to include individual components of the reward

    # # Stacked plot
    # ax1.set_xlim(0, xlim_max)
    # ax1.set_ylim(ylim_min_1, 0)
    # ax1.stackplot(np.linspace(0, frame, frame + 1),
    #               np.cumsum(rewards_imbalance_cost[:frame + 1]), np.cumsum(rewards_fuel_cost[:frame + 1]),
    #               labels=['imbalance', 'fuel'])
    # ax1.plot(generator_2_levels[:frame])
    # ax1.set_xlabel("time")
    # ax1.set_ylabel("cumulative reward")

    # generator levels
    ax2.set_xlim(0, xlim_max)
    ax2.set_ylim(0.4, 3.1)
    ax2.plot(generator_1_levels[:frame])
    ax2.plot(generator_2_levels[:frame])
    ax2.set_xlabel("time")
    ax2.set_ylabel("generator levels")

    actions
    ax3.set_xlim(0, xlim_max)
    ax3.set_ylim(ylim_min_3, ylim_max_3)
    ax3.plot(actions[:frame])
    ax3.set_xlabel("time")
    ax3.set_ylabel("actions")

    # # agent predictions simple
    # ax3.set_ylim(ylim_min_4, ylim_max_4)
    # ax3.set_xlim(0, xlim_max)
    # ax3.plot(agent_predictions[frame])
    # ax3.set_xlabel("time")
    # ax3.set_ylabel("predictions")

    # # agent prediction simple
    # ax4.set_ylim(ylim_min_4, ylim_max_4)
    # ax4.set_xlim(0, xlim_max)
    # ax4.plot(agent_prediction[frame], 'b')
    # ax4.set_xlabel("time")
    # ax4.set_ylabel("prediction")

    # agent prediction
    ax4.set_ylim(ylim_min_4, ylim_max_4)
    ax4.set_xlim(0, xlim_max)
    ax4.plot(agent_prediction[frame, :frame + 1], 'b')  # up to and including current time
    ax4.plot(np.linspace(frame, 96, num=int(96-frame) , dtype=int), agent_prediction[frame, frame:], 'b', alpha=0.35)
    ax4.plot(generator_1_levels[:frame] + generator_2_levels[:frame], 'r', label='generator_levels')
    ax4.set_xlabel("time")
    ax4.set_ylabel("prediction")


    # Repack ax
    return ((ax1, ax2), (ax3, ax4))



def plot_picture(state, fname):
    rewards_total = np.array(state.rewards_all)
    # rewards_fuel_cost = np.array(state.rewards_fuel_cost)
    # rewards_imbalance_cost = np.array(state.rewards_imbalance_cost)
    generator_1_levels = np.array(state.generator_1_levels_all)
    generator_2_levels = np.array(state.generator_2_levels_all)
    actions = np.array(state.actions_all)
    agent_predictions = np.array(state.agent_predictions_all)
    agent_prediction = get_agent_prediction(agent_predictions)
    xlim_max = np.shape(rewards_total)[0]
    ylim_min_1 = 1.03 * np.sum(rewards_total)
    ylim_max_3 = 1.1 * np.amax(actions)
    ylim_min_3 = 0.9 * np.amin(actions)
    ylim_max_4 = 1.05 * np.amax(agent_predictions)
    ylim_min_4 = 1.05 * np.amin(agent_predictions)

    J = 2
    K = 2
    fig, ax = plt.subplots(J, K)
    ((ax1, ax2), (ax3, ax4)) = ax
    # cumulative total cost
    ax1.set_xlim(0, xlim_max)
    ax1.set_ylim(ylim_min_1, 0)
    ax1.plot(np.cumsum(rewards_total))
    ax1.set_xlabel("time")
    ax1.set_ylabel("cumulative reward")
    # could be expanded to include individual components of the reward

    # # Stacked plot
    # ax1.set_xlim(0, xlim_max)
    # ax1.set_ylim(ylim_min_1, 0)
    # ax1.stackplot(np.linspace(0, frame, frame + 1),
    #               np.cumsum(rewards_imbalance_cost[:frame + 1]), np.cumsum(rewards_fuel_cost[:frame + 1]),
    #               labels=['imbalance', 'fuel'])
    # ax1.plot(generator_2_levels[:frame])
    # ax1.set_xlabel("time")
    # ax1.set_ylabel("cumulative reward")

    # generator levels
    ax2.set_xlim(0, xlim_max)
    ax2.set_ylim(0.4, 3.1)
    ax2.plot(generator_1_levels)
    ax2.plot(generator_2_levels)
    ax2.set_xlabel("time")
    ax2.set_ylabel("generator levels")

    actions
    ax3.set_xlim(0, xlim_max)
    ax3.set_ylim(ylim_min_3, ylim_max_3)
    ax3.plot(actions)
    ax3.set_xlabel("time")
    ax3.set_ylabel("actions")

    # # agent predictions simple
    # ax3.set_ylim(ylim_min_4, ylim_max_4)
    # ax3.set_xlim(0, xlim_max)
    # ax3.plot(agent_predictions[frame])
    # ax3.set_xlabel("time")
    # ax3.set_ylabel("predictions")

    # # agent prediction simple
    # ax4.set_ylim(ylim_min_4, ylim_max_4)
    # ax4.set_xlim(0, xlim_max)
    # ax4.plot(agent_prediction[frame], 'b')
    # ax4.set_xlabel("time")
    # ax4.set_ylabel("prediction")

    # agent prediction
    ax4.set_ylim(ylim_min_4, ylim_max_4)
    ax4.set_xlim(0, xlim_max)
    ax4.plot(agent_prediction[-1], 'b')  # up to and including current time
    ax4.plot(generator_1_levels + generator_2_levels, 'r', label='generator_levels')
    ax4.set_xlabel("time")
    ax4.set_ylabel("prediction")

    plt.savefig(fname)


def plot_video(state, fname):
    rewards_total = np.array(state.rewards_all)
    # rewards_fuel_cost = np.array(state.rewards_fuel_cost)
    # rewards_imbalance_cost = np.array(state.rewards_imbalance_cost)
    generator_1_levels = np.array(state.generator_1_levels_all)
    generator_2_levels = np.array(state.generator_2_levels_all)
    actions = np.array(state.actions_all)
    agent_predictions = np.array(state.agent_predictions_all)
    agent_prediction = get_agent_prediction(agent_predictions)
    xlim_max = np.shape(rewards_total)[0]
    ylim_min_1 = 1.03 * np.sum(rewards_total)
    ylim_max_3 = 1.1 * np.amax(actions)
    ylim_min_3 = 0.9 * np.amin(actions)
    ylim_max_4 = 1.05 * np.amax(agent_predictions)
    ylim_min_4 = 1.05 * np.amin(agent_predictions)
    state_tuple = (rewards_total,
     # rewards_fuel_cost,
     # rewards_imbalance_cost,
     generator_1_levels,
     generator_2_levels,
     actions,
     agent_predictions,
     agent_prediction)
    lim_tuple = (xlim_max, ylim_min_1, ylim_max_3, ylim_min_3, ylim_max_4, ylim_min_4)
    J = 2
    K = 2
    fig, ax = plt.subplots(J, K)

    def animate(i):
        print('{} frame {} rendered'.format(fname, i))
        # Clear the axis
        for j in range(J):
            for k in range(K):
                ax[j, k].clear()
        # Plot on fresh axis
        plot_frame(state_tuple, lim_tuple, ax, frame=i)
        plt.tight_layout()
        plt.show()
        return None

    ani = animation.FuncAnimation(
        fig, animate, frames=96, interval=100)
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=800)
    ani.save('{}.mp4'.format(fname), writer=writer, dpi=300)


def plot2(state, fname="episode"):
    plot_video(state, fname)


def plot3(state, fname="episode"):
    plot_picture(state, fname)
