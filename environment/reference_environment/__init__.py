from gym.envs.registration import register

register(
    id="rangl-nztc-v0",
    entry_point="reference_environment.env:GymEnv",
)
