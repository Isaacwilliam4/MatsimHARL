from gymnasium.envs.registration import register

#registration for the optimal charging placement environment 
register(
    id="ocp", 
    entry_point="harl.envs.rl_ocp_env:RLOCPEnv",
)
