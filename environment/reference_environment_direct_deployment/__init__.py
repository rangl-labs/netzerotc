from gym.envs.registration import register

register(
    id="reference-environment-direct-deployment-v0", entry_point="reference_environment_direct_deployment.env:GymEnv",
)
