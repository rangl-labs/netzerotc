from gym.envs.registration import register

register(
    id="nztc-open-loop-v0",
    entry_point="open_loop_env.env:GymEnv",
)
