import logging

import gym
import reference_environment

from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Checks

# Create an environment named env
env = gym.make("reference_environment:reference-environment-v0")
# Generate a random action and check it has the right length
action = env.action_space.sample()
assert len(action) == 2

# Reset the environment
env.reset()
# Check the to_observation method
assert len(env.observation_space.sample()) == len(env.state.to_observation())
done = False
while not done:
    # Specify the action. Check the effect of any fixed policy by specifying the action here:
    action = [4, 5] # env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    logger.debug(f"step_count: {env.state.step_count}")
    logger.debug(f"action: {action}")
    logger.debug(f"observation: {observation}")
    logger.debug(f"reward: {reward}")
    logger.debug(f"done: {done}")
    print()

# Plot the episode
env.plot("fixed_policy.png")
assert Path("fixed_policy.png").is_file()

# logger.info(f"observations_all: {env.state.observations_all}")
# logger.info(f"actions_all: {env.state.actions_all}")
# logger.info(f"rewards_all: {env.state.rewards_all}")

assert len(env.state.observations_all) > 0
assert len(env.state.actions_all) > 0
assert len(env.state.rewards_all) > 0
# Check episode has desired number of steps
assert env.state.observations_all[-1][0] == env.param.steps_per_episode - 1

# check that specifying the same seed gives the same noise
env.seed(123)
obs1 = env.reset()
env = gym.make("reference_environment:reference-environment-v0")
env.seed(123)
obs2 = env.reset()
assert obs1 == obs2

# check that the seed can be reverted to None, so reset() gives different noise
env = gym.make("reference_environment:reference-environment-v0")
env.seed(123)
env.seed(None)
obs1 = env.reset()
obs2 = env.reset()
assert not obs1 == obs2


