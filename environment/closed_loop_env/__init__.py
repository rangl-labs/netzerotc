from gym.envs.registration import register

register(
    id="nztc-closed-loop-v0",
    entry_point="closed_loop_env.env:GymEnv",
)
