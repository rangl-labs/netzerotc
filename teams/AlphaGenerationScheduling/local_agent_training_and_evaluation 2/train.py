#!/usr/bin/env python
import gym
import numpy as np
from util import (Trainer, ObservationTransform,
                  HorizonObservationWrapper, PhaseRewardWrapper,
                  RandomActionWrapper, JoesActionWrapper, OurActionWrapper)
from stable_baselines3 import PPO
from gym import spaces, ActionWrapper


######################################################################
# This section includes all of the testing we performed with wrappers.
######################################################################


# env = gym.make("reference_environment:reference-environment-v0")

# Train an RL agent on the environment
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=20000)


# ## Training on horizon observations
# env = HorizonObservationWrapper(gym.make("reference_environment:reference-environment-v0"),
#                               horizon_length=30,
#                               transform_name="Standard")
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=20000)

### Testing random action wrapper
# env = JoesActionWrapper(gym.make("reference_environment:reference-environment-v0"))
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=20000)

### Testing phase reward wrapper
# env=PhaseRewardWrapper(gym.make("reference_environment:reference-environment-v0"), phase="Peak")
# trainer = Trainer(env)
# trainer.train_rl(models_to_train=1, episodes_per_model=1000)


### Test nested wrappers
# env_horizon = HorizonObservationWrapper(gym.make("reference_environment:reference-environment-v0"),
#                               horizon_length=20,
#                               transform_name="Standard")
# env = PhaseRewardWrapper(env_horizon, phase="Peak")
# trainer = Trainer(env)
# # trainer.train_rl(models_to_train=1,episodes_per_model=1000)                   # Begin Training
# trainer.retrain_rl(model=PPO.load("logs/best_model_peak_20"), episodes=1000)   # Retraining


### Test training on peak then full
# env_horizon = HorizonObservationWrapper(gym.make("reference_environment:reference-environment-v0"),
#                               horizon_length=30,
#                               transform_name="Standard")
# env_peak = PhaseRewardWrapper(env_horizon, phase="Peak")          # Set Phase to Peak
# trainer = Trainer(env_peak)
# trainer.train_rl(models_to_train=1,episodes_per_model=3000)       # Begin Training
# model = PPO.load("logs/best_model_peak_30")                               # Load best model
# model.learning_rate = 0.0003
# env_full = PhaseRewardWrapper(env_horizon, phase="Full")          # Set Phase to Full
# trainer = Trainer(env_full)
# trainer.retrain_rl(model=model, episodes=50000)                    # Re-train on full phase

### Test Retraining
# env=HorizonObservationWrapper(gym.make("reference_environment:reference-environment-v0"),
#                               horizon_length=30,
#                               transform_name="Deltas")
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import MlpPolicy
# # from stable_baselines3.common.callbacks import EvalCallback
# # eval_callback = EvalCallback(env, best_model_save_path='./logs/',
# #                              log_path='./logs/', eval_freq=500,
# #                              deterministic=True, render=False)
# # model = PPO.load("logs/best_model_full_30")
# model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="./logs/",
#             gamma=1,
#             learning_rate=0.0003,
#             )
#

### Test training on peak then full
# env_action = OurActionWrapper(gym.make("reference_environment:reference-environment-v0"))
# env_horizon = HorizonObservationWrapper(env_action,
#                               horizon_length=30,
#                               transform_name="Standard")
# env_peak = PhaseRewardWrapper(env_horizon, phase="Peak")          # Set Phase to Peak
# env_full = PhaseRewardWrapper(env_horizon, phase="Full")          # Set Phase to Full
#
#
# trainer = Trainer(env_peak)
# # trainer.train_rl(models_to_train=1,episodes_per_model=3000)       # Begin Training
# model = PPO.load("logs/best_model_peak_30")                               # Load best model
# model.learning_rate = 0.0003
# env_full = PhaseRewardWrapper(env_horizon, phase="Full")          # Set Phase to Full
# trainer = Trainer(env_full)
# trainer.retrain_rl(model=model, episodes=50000)                    # Re-train on full phase


#################################################################
# This section below is how the submitted model has been trained.
#################################################################


### Testing DDPG ###
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback

### Wrap the reference environment for training
# Action wrapper to specify correct action space
env_action = OurActionWrapper(gym.make("reference_environment:reference-environment-v0"))

# Obseration wrapper to transform observations to horizon of fixed length.
env_horizon = HorizonObservationWrapper(env_action,
                              horizon_length=7,
                              transform_name="Standard")

# Wrapper for changing phase to full or peak  - this might be a good suggestion to rangl, instead of uncommenting
# code in the reference environment?
env = PhaseRewardWrapper(env_horizon, phase="Full")          # Set Phase to Full

### DDPG Noise
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

### Set the policy and the DDPG noise
model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./logs/",
            gamma=0.99,
            learning_rate=0.0003,
            )

### Train the specified model
trainer = Trainer(env)
trainer.retrain_rl(model, episodes=10000)



