from gym.envs.registration import register

register(
    id="nztc-open-loop-v0",
    entry_point="rangl.env_open_loop.env:GymEnv",
)


register(
    id="nztc-closed-loop-v0",
    entry_point="rangl.env_closed_loop.env:GymEnv",
)
