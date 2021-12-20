import os

from util import Client

# when running the agent locally, assume that the environment is accesible at localhost:5000
# when running a containerised agent, assume that the environment is accesible at $RANGL_ENVIRONMENT_URL (typically http://nztc:5000)
remote_base = os.getenv("RANGL_ENVIRONMENT_URL", "http://localhost:5000/")

client = Client(remote_base)

env_id = "nztc-dev-v0"
seed = int(os.getenv("RANGL_SEED", 123456))
instance_id = client.env_create(env_id, seed)


client.env_monitor_start(
    instance_id,
    directory=f"monitor/{instance_id}",
    force=True,
    resume=False,
    video_callable=False,
)

client.env_reset(instance_id)

while True:
    action = client.env_action_space_sample(instance_id)
    observation, reward, done, info = client.env_step(instance_id, action)
    print(instance_id, reward)
    if done:
        print(instance_id)
        break

client.env_monitor_close(instance_id)

print("done", done)


# make sure you print the instance_id as the last line in the script
print(instance_id)
