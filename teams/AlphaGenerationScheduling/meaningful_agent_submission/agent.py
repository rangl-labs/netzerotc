import os

from stable_baselines3 import DDPG

from util import Client

# when running the agent locally, assume that the environment is accesible at localhost:5000
# when running a containerised agent, assume that the environment is accesible at $RANGL_ENVIRONMENT_URL (typically http://environment:5000)
remote_base = os.getenv("RANGL_ENVIRONMENT_URL", "http://localhost:5000/")

client = Client(remote_base)

env_id = "reference-environment-v0"
seed = int(os.getenv("RANGL_SEED", 123456))

instance_id = client.env_create(env_id, seed)


client.env_monitor_start(
    instance_id,
    directory=f"monitor/{instance_id}",
    force=True,
    resume=False,
    video_callable=False,
)

model = DDPG.load("MODEL_ALPHA_GENERATION.zip")

observation = client.env_reset(instance_id)

print(observation)

import numpy as np
def ObservationTransform(obs, H, transform, steps_per_episode=int(96)):
    step_count, generator_1_level, generator_2_level = obs[:3]
    agent_prediction = np.array(obs[3:])

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

while True:

    # Perform observation transform to pass mapped obs to agent
    observation = ObservationTransform(observation, H=7, transform="Standard")

    action, _ = model.predict(observation, deterministic=True)
    action = [float(action[0]), float(action[1])]
    observation, reward, done, info = client.env_step(instance_id, action)
    print(instance_id, reward)
    if done:
        print(instance_id)
        break


client.env_monitor_close(instance_id)

print("done", done)

# make sure you print the instance_id as the last line in the script
print(instance_id)
