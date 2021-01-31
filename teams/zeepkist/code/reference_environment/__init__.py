from gym.envs.registration import register

register(
    id="reference-environment-v0", entry_point="reference_environment.env:GymEnv",
)
