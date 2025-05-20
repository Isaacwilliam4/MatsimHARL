import numpy as np
import torch
from pathlib import Path
from harl.envs.ocp.chargers import *
from gymnasium.spaces import Box
from gymnasium import spaces
from harl.envs.ocp.matsim_gnn import MatsimGNN
from harl.envs.ocp.matsim_xml_dataset import MatsimXMLDataset
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
                 charge_model: MatsimGNN,
                 charge_model_loop: int = 100,
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
        self.charge_model = charge_model
        self.iteration = 0
        self.charge_model_loop = charge_model_loop

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.charge_model.parameters(), lr=1e-3)

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



    def save_charger_config_to_csv(self, dir):
        """
        Save the current charger configuration to a CSV file.

        Args:
            csv_path (str): Path to save the CSV file.
        """
        static_chargers = []
        dynamic_chargers = []
        charger_config = self.dataset.graph.edge_attr[:, -self.dataset.num_charger_types:]

        for idx, row in enumerate(charger_config):
            if not row[0]:
                if row[1]:
                    dynamic_chargers.append(int(self.dataset.edge_mapping.inverse[idx]))
                elif row[2]:
                    static_chargers.append(int(self.dataset.edge_mapping.inverse[idx]))

        df = pd.DataFrame(
            {
                "reward": [self.reward],
                "cost": [self.dataset.charger_cost.item()],
                "static_chargers": [static_chargers],
                "dynamic_chargers": [dynamic_chargers],
            }
        )
        df.to_csv(Path(dir / "best_chargers.csv"), index=False)

    def step(self, actions):
        """
        Take an action and return the next state, reward, done, and info.

        Args:
            actions (list): Actions to perform.

        Returns:
            tuple: Next state, reward, done flags, and additional info.
        """
        self.dataset.create_chargers_xml_gymnasium(
            actions
        )
        charger_cost = self.dataset.parse_charger_network_get_charger_cost()
        charger_cost_reward = (charger_cost / self.dataset.max_charger_cost).item()

        self.iteration += 1
        loss = None
        if self.iteration % self.charge_model_loop == 0:
            avg_charge_reward, _ = self.send_reward_request() 
            self.charge_model.train()
            x, edge_index = self.dataset.linegraph.x.to(self.device), self.dataset.linegraph.edge_index.to(self.device) 
            output = self.charge_model(x, edge_index)
            target = torch.tensor(avg_charge_reward).to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            self.optimizer.step()
            self.charge_model.eval()
        else:
            avg_charge_reward = self.charge_model(self.dataset.linegraph.x.to(self.device), self.dataset.linegraph.edge_index.to(self.device))
        
        _reward = avg_charge_reward - charger_cost_reward

        self.reward = _reward
        if _reward > self.best_reward:
            self.best_reward = _reward

        return (
            self.dataset.linegraph.x.flatten(),
            _reward,
            self.done,
            self.done,
            dict(graph_env_inst=self, 
                 avg_charge_reward=avg_charge_reward, 
                 charger_cost_reward=charger_cost_reward,
                 charge_model_loss = None if loss is None else loss.item()),
        )

    def seed(self, seed: int):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def split(self, a):
        return [a[i] for i in range(self.n_agents)]
