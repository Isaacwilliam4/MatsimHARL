from gymnasium.envs.registration import register

register(
    id="FlowSimEnv_v0", 
    entry_point="harl.envs.flowsim.flowsim:FlowSimEnv",
)