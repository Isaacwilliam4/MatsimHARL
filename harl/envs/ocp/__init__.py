from gymnasium.envs.registration import register

#registration for the optimal charging placement environment 
register(
    id="ocp", 
    entry_point="harl.envs.ocp:RLOCPEnv",
)
