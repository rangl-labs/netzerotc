import os

from stable_baselines3 import DDPG

from util import Client


ENV_ID = "nztc-closed-loop-v0"
MODEL_PATH = "saved_models/MODEL_closed_loop_0"

# when running the agent locally, assume that the environment is accesible at localhost:5000
# when running a containerised agent, assume that the environment is accesible at $RANGL_ENVIRONMENT_URL (typically http://nztc:5000)
remote_base = os.getenv("RANGL_ENVIRONMENT_URL", "http://localhost:5000/")

client = Client(remote_base)

seed = int(os.getenv("RANGL_SEED", 123456))
instance_id = client.env_create(ENV_ID, seed)


client.env_monitor_start(
    instance_id,
    directory=f"monitor/{instance_id}",
    force=True,
    resume=False,
    video_callable=False,
)

obs = client.env_reset(instance_id)

model = DDPG.load(MODEL_PATH)

while True:
    action, _ = model.predict(obs, deterministic=True)
    action = [float(action[0]), float(action[1]), float(action[2])]
    obs, reward, done, info = client.env_step(instance_id, action)
    print(instance_id, reward)
    if done:
        print(instance_id)
        break

client.env_monitor_close(instance_id)

print("done", done)


# make sure you print the instance_id as the last line in the script
print(instance_id)
