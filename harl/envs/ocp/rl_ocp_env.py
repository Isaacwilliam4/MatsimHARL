import numpy as np
import torch
from pathlib import Path
from rlevmatsim.envs.ocp.chargers import *
from gymnasium.spaces import Box
from gymnasium import spaces
from rlevmatsim.envs.ocp.matsim_gnn import MatsimGNN
from rlevmatsim.envs.ocp.matsim_mlp import MatsimMLP
from rlevmatsim.envs.ocp.matsim_xml_dataset import MatsimXMLDataset
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import zipfile
import json
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy as dc

class RLOCPEnv:
    """
    A custom Gymnasium environment for Matsim graph-based simulations.
    """

    def __init__(self, 
                 dataset: MatsimXMLDataset,
                 device: str,
                 **kwargs):
        """
        Initialize the environment.

        Args:
            network_path (str): Path to the configuration file.
            n_agents (int): Number of agents in the environment.
            save_dir (str): Directory to save outputs.
        """

        # each agent monitors a cluster and determines the chargers that should go there
        self.device = device
        self.dataset = dataset.copy()
        self.n_agents = self.dataset.num_clusters
        self.iteration = 0

        self.action_space = self.repeat(
            spaces.MultiDiscrete([self.dataset.num_charger_types]*dataset.max_cluster_len)
        )

        self.observation_space : Box = self.repeat(
            Box(
                low=0,
                high=1,
                shape=(self.dataset.linegraph.x.numel(),)
            )
        )

        self.share_observation_space : Box = self.repeat(
            Box(
                low=0,
                high=1,
                shape=(self.dataset.linegraph.x.numel(),)
            )
        )

        self.reward: float = 0
        self.best_reward = -np.inf

    def reset(self, **kwargs):
        """
        Reset the environment to its initial state.

        Returns:
            np.ndarray: Initial state of the environment.
            dict: Additional information.
        """
        return  self.repeat(self.dataset.linegraph.x.flatten()), self.repeat(self.dataset.linegraph.x.flatten()), None

    def step(self, actions):
        """
        Take an action and return the next state, reward, done, and info.

        Args:
            actions (list): Actions to perform.

        Returns:
            tuple: Next state, reward, done flags, and additional info.
        """

        curr_charger_config = self.dataset.linegraph.x[:, -self.dataset.num_charger_types:]
        new_charger_config = torch.zeros_like(curr_charger_config)
        new_charger_config[torch.arange(new_charger_config.shape[0]), actions] = 1
        self.dataset.linegraph.x[:, -self.dataset.num_charger_types:] = new_charger_config
        
        charger_cost_reward = self.dataset.get_charger_cost_reward()
        self.iteration += 1

        if self.iteration % self.dataset.charge_model_loop == 0:
            self.dataset.train_charge_model(self.dataset.charge_model_iters)

        if isinstance(self.dataset.charge_model, MatsimGNN):
            x, edge_index = self.dataset.linegraph.x.to(self.dataset.device), self.dataset.linegraph.edge_index.to(self.dataset.device) 
            charge_reward = self.dataset.charge_model(x, edge_index)
        elif isinstance(self.dataset.charge_model, MatsimMLP):
            x = self.dataset.linegraph.x.to(self.dataset.device)
            charge_reward = self.dataset.charge_model(x)

        self.charge_reward = charge_reward.detach().item()
        _reward = (charge_reward - charger_cost_reward).detach().item()

        self.reward = _reward
        if _reward > self.best_reward:
            self.best_reward = _reward

        return (
            self.repeat(self.dataset.linegraph.x.flatten()),
            self.repeat(self.dataset.linegraph.x.flatten()),
            self.repeat(_reward),
            self.repeat(False),
            self.repeat(dict(graph_env_inst=self, 
                 charge_reward=charge_reward.detach().item(), 
                 charger_cost_reward=charger_cost_reward))
            ,
            None
        )

    def seed(self, seed: int):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def split(self, a):
        return [a[i] for i in range(self.n_agents)]
