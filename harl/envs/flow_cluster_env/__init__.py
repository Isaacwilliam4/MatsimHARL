from gymnasium.envs.registration import register

register(
    id="ClusterFlowMatsimGraphEnv-v0", 
    entry_point="cluster_flow_matsim_graph_env:ClusterFlowMatsimGraphEnv",
)