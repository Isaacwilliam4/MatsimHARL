import xml.etree.ElementTree as ET
import torch
import shutil
from torch_geometric.data import Dataset
from torch_geometric.transforms import LineGraph
from gymnasium import spaces
from torch_geometric.data import Data
from pathlib import Path
from bidict import bidict
from rlevmatsim.envs.ocp.chargers import Charger, NoneCharger, StaticCharger, DynamicCharger
from rlevmatsim.envs.ocp.matsim_gnn import MatsimGNN
from rlevmatsim.envs.ocp.matsim_mlp import MatsimMLP
from sklearn.cluster import KMeans
import numpy as np
import os
from datetime import datetime
from copy import deepcopy as dc
import zipfile
import json
import requests
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

class MatsimXMLDataset(Dataset):
    """
    A dataset class for parsing MATSim XML files and creating a graph
    representation using PyTorch Geometric.
    """

    def __init__(
        self,
        config_path: Path,
        num_clusters: int,
        device: str,

    ):
        """
        Initializes the MatsimXMLDataset.

        Args:
            config_path (Path): Path to the MATSim configuration file.
            time_string (str): Unique identifier for temporary directories.
        """
        super().__init__(transform=None)
        self.device = device


        time_string = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.setup_dirs(config_path, time_string)
        
        self.num_clusters = num_clusters
        self.charger_cost = 0
        self.charger_model_loss = None


        self.node_mapping: bidict[str, int] = (
            bidict()
        )  #: Store mapping of node IDs to indices in the graph

        self.edge_mapping: bidict[str, int] = (
            bidict()
        )  #: (key:edge id, value: index in edge list)
        self.edge_attr_mapping: bidict[str, int] = (
            bidict()
        )  #: key: edge attribute name, value: index in edge attribute list
        self.graph: Data = Data()
        self.linegraph_transform = LineGraph()
        self.charger_list = [NoneCharger, StaticCharger, DynamicCharger]
        self.num_charger_types = len(self.charger_list)
        self.max_charger_cost = 0
        self.create_edge_attr_mapping()
        self.parse_matsim_network()
        self.get_charger_cost_reward()
    
    def init_model(self, model: MatsimGNN, charge_model_loop: int, charge_model_iters: int, lr: float):
        self.charge_model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.charge_model.parameters(), lr=lr)
        self.charge_model_loop = charge_model_loop
        self.charge_model_iters = charge_model_iters

    def train_charge_model(self, iterations):
        self.charge_model.train()
        pbar = tqdm(range(iterations), desc="Training Sim Learner")
        for _ in pbar:
            self.sample_chargers()
            if isinstance(self.charge_model, MatsimGNN):
                x, edge_index = self.linegraph.x.to(self.device), self.linegraph.edge_index.to(self.device) 
                output = self.charge_model(x, edge_index)
            elif isinstance(self.charge_model, MatsimMLP):
                x = self.linegraph.x.to(self.device)
                output = self.charge_model(x)
            response = self.send_reward_request()
            target = torch.tensor(response[0]).to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            self.charger_model_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix(loss=loss.item())
            self.charger_model_loss = loss.item()
            
        self.charge_model.eval()

    def setup_dirs(self, config_path, time_string):
        self.time_string = time_string
        tmp_dir = Path("/tmp/" + time_string)
        output_path = Path(tmp_dir / "output")

        shutil.copytree(config_path.parent, tmp_dir)

        self.config_path = Path(tmp_dir / config_path.name)

        (
            network_file_name,
            plans_file_name,
            vehicles_file_name,
            chargers_file_name,
            _
        ) = self.setup_config(self.config_path, str(output_path))

        self.charger_xml_path = Path(tmp_dir / chargers_file_name)
        self.network_xml_path = Path(tmp_dir / network_file_name)
        self.plan_xml_path = Path(tmp_dir / plans_file_name)
        self.vehicle_xml_path = Path(tmp_dir / vehicles_file_name)
        self.consumption_map_path = Path(tmp_dir / "consumption_map.csv")

    def copy(self):
        time_string = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        res = dc(self)
        res.setup_dirs(self.config_path, time_string)
        return res

    def len(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.graph)
    
    def save_charger_config_to_csv(self, dir, reward):
        """
        Save the current charger configuration to a CSV file.

        Args:
            csv_path (str): Path to save the CSV file.
        """
        static_chargers = []
        dynamic_chargers = []
        charger_config = self.graph.edge_attr[:, -self.num_charger_types:]

        for idx, row in enumerate(charger_config):
            charger_idx = torch.nonzero(row)[0]
            charger_type = self.charger_list[charger_idx]
            if charger_type.type == "none":
                continue
            elif charger_type.type == "default":
                static_chargers.append(int(self.edge_mapping.inverse[idx]))
            elif charger_type.type == "dynamic":
                dynamic_chargers.append(int(self.edge_mapping.inverse[idx]))

        df = pd.DataFrame(
            {
                "reward": [reward],
                "cost": [self.charger_cost],
                "static_chargers": [static_chargers],
                "dynamic_chargers": [dynamic_chargers],
            }
        )
        df.to_csv(Path(dir) / "best_chargers.csv", index=False)

    def _min_max_normalize(self, tensor, reverse=False):
        """
        Normalizes or denormalizes a tensor using min-max scaling.

        Args:
            tensor (Tensor): The tensor to normalize or denormalize.
            reverse (bool): Whether to reverse the normalization. Default
                is False.

        Returns:
            Tensor: The normalized or denormalized tensor.
        """
        if reverse:
            return tensor * (self.max_mins[1] - self.max_mins[0]) + self.max_mins[0]
        return (tensor - self.max_mins[0]) / (self.max_mins[1] - self.max_mins[0])
    
    def setup_config(self, config_xml_path, output_dir, num_iterations=0):
        """
        Configures MATSim XML file with iterations and output directory.

        Args:
            config_xml_path (str): Path to the config XML file.
            output_dir (str): Directory for MATSim results.
            num_iterations (int): Number of MATSim iterations to run.

        Returns:
            tuple: Paths to network, plans, vehicles, and charger XML files.
        """
        tree = ET.parse(config_xml_path)
        root = tree.getroot()

        network_file, plans_file, vehicles_file, chargers_file, counts_file = None, None, None, None, None

        for module in root.findall(".//module"):
            for param in module.findall("param"):
                if param.get("name") == "lastIteration":
                    param.set("value", str(num_iterations))
                if param.get("name") == "outputDirectory":
                    param.set("value", output_dir)
                if param.get("name") == "inputNetworkFile":
                    network_file = param.get("value")
                if param.get("name") == "inputPlansFile":
                    plans_file = param.get("value")
                if param.get("name") == "vehiclesFile":
                    vehicles_file = param.get("value")
                if param.get("name") == "chargersFile":
                    chargers_file = param.get("value")
                if param.get("name") == "inputCountsFile":
                    counts_file = param.get("value")

        with open(config_xml_path, "wb") as f:
            f.write(b'<?xml version="1.0" ?>\n')
            f.write(
                b'<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">\n'
            )
            tree.write(f)

        return network_file, plans_file, vehicles_file, chargers_file, counts_file

    def save_server_output(self, dir, response, filename):
        """
        Save server output to a zip file and extract its contents.

        Args:
            response (requests.Response): Server response object.
            filename (str): name of file.
        """
        zip_filename = Path(dir, f"{filename}.zip")
        extract_folder = Path(dir, filename)

        # Use a lock to prevent simultaneous access
        # lock = FileLock(lock_file)

        # with lock:
        # Save the zip file
        with open(zip_filename, "wb") as f:
            f.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    def save_output(self, dir, filename):
        self.create_chargers_from_graph()
        _, response = self.send_reward_request()
        self.save_server_output(dir, response, filename)

    def send_reward_request(self):
        """
        Send a reward request to the server and process the response.

        Returns:
            tuple: Reward value and server response.
        """
        url = "http://localhost:8000/getReward"
        files = {
            "config": open(self.config_path, "rb"),
            "network": open(self.network_xml_path, "rb"),
            "plans": open(self.plan_xml_path, "rb"),
            "vehicles": open(self.vehicle_xml_path, "rb"),
            "chargers": open(self.charger_xml_path, "rb"),
            "consumption_map": open(self.consumption_map_path, "rb"),
        }
        response = requests.post(
            url, params={"folder_name": self.time_string}, files=files
        )
        json_response = json.loads(response.headers["X-response-message"])
        reward = json_response["reward"]

        return float(reward), response
    
    def create_edge_attr_mapping(self):
        """
        Creates a mapping of edge attributes to their indices.
        """
        self.edge_attr_mapping = bidict({"length": 0, "freespeed": 1, "capacity": 2, "slopes":3})
        edge_attr_idx = len(self.edge_attr_mapping)
        for charger in self.charger_list:
            self.edge_attr_mapping[charger.type] = edge_attr_idx
            edge_attr_idx += 1

    def parse_matsim_network(self):
        """
        Parses the MATSim network XML file and creates a graph representation.
        """
        tree = ET.parse(self.network_xml_path)
        root = tree.getroot()
        matsim_node_ids = []
        node_ids = []
        node_pos = []
        edge_index = []
        edge_attr = []
        node_coords_list = []
        self.node_coords = {}
        self.clusters = {}
        node_idx_to_link_idx = {}

        for i, node in enumerate(root.findall(".//node")):
            node_id = node.get("id")
            matsim_node_ids.append(node_id)
            node_pos.append([float(node.get("x")), float(node.get("y"))])
            self.node_mapping[node_id] = i
            node_ids.append(i)
            curr_x = float(node.get("x"))
            curr_y = float(node.get("y"))
            node_coords_list.append([curr_x, curr_y])
            self.node_coords[node_id] = (curr_x, curr_y)

        tot_attr = len(self.edge_attr_mapping)

        for i, link in enumerate(root.findall(".//link")):
            from_node = link.get("from")
            to_node = link.get("to")
            from_idx = self.node_mapping[from_node]
            to_idx = self.node_mapping[to_node]
            edge_index.append([from_idx, to_idx])
            curr_link_attr = torch.zeros(tot_attr)
            self.edge_mapping[link.get("id")] = i
            if from_idx in node_idx_to_link_idx:
                node_idx_to_link_idx[from_idx].append(i)
            else:
                node_idx_to_link_idx[from_idx] = [i]

            for key, value in self.edge_attr_mapping.items():
                if key in link.attrib:
                    if key == "length":
                        """
                        Add the cost of either the static charger or the 
                        dynamic charger times the length of the link, 
                        converted to km from m.
                        """
                        link_len_km = float(link.get(key)) * 0.001
                        self.max_charger_cost += max(
                            StaticCharger.price,
                            DynamicCharger.price * link_len_km,
                        )
                    curr_link_attr[value] = float(link.get(key))

            edge_attr.append(curr_link_attr)

        self.graph.x = torch.tensor(node_ids).view(-1, 1)
        self.graph.pos = torch.tensor(node_pos)
        self.graph.edge_index = torch.tensor(edge_index).t()
        self.graph.edge_attr = torch.stack(edge_attr)
        self.linegraph = self.linegraph_transform(self.graph)
        self.max_mins = torch.stack(
            [
                torch.min(self.graph.edge_attr[:, :-self.num_charger_types], dim=0).values,
                torch.max(self.graph.edge_attr[:, :-self.num_charger_types], dim=0).values,
            ]
        )

        self.graph.edge_attr[:, :-self.num_charger_types] = self._min_max_normalize(
            self.graph.edge_attr[:, :-self.num_charger_types]
        )
        self.state = self.graph.edge_attr

        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(np.array(node_coords_list))
        self.kmeans = kmeans

        for idx, label in enumerate(kmeans.labels_):
            cluster_id = label
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            for edge_idx in node_idx_to_link_idx[idx]:
                self.clusters[cluster_id].append(edge_idx)

        self.clusters = {k: v for k,v in sorted(self.clusters.items(), key=lambda x: x[0])}
        self.available_actions = []
        self.max_cluster_len = 0
        
        for cluster in self.clusters.values():
            n = len(cluster)
            self.available_actions.append(np.arange(n))
            if n > self.max_cluster_len:
                self.max_cluster_len = n

    def save_clusters(self, dir):
        filepath = Path(Path(dir) / "clusters.txt")
        if not os.path.exists(filepath.parent):
            os.makedirs(filepath.parent)
        with open(filepath, "w") as f:
            for cluster_id, edges in self.clusters.items():
                f.write(f"{cluster_id}:")
                for edge_idx in edges:
                    f.write(f"{self.edge_mapping.inv[edge_idx]},")
                f.write('\n')

    def sample_chargers(
        self,
    ):
        """
        Create a chargers XML file for MATSim using a multi-discrete action space.

        Args:
            charger_xml_path (Path): Path to save the chargers XML file.
            charger_list (list): List of charger type objects.
            actions (spaces.MultiDiscrete): Action space with dimension (num_edges),
                where each value corresponds to the index of the charger list
                (0 is no charger).
            link_id_mapping (bidict): Mapping of link IDs to indices.
        """
        chargers = ET.Element("chargers")
        actions = torch.randint(0, 3, size=(self.linegraph.num_nodes,))
        new_charger_config = torch.zeros_like(self.linegraph.x[:,-self.num_charger_types:])
        new_charger_config[torch.arange(new_charger_config.shape[0]), actions] = 1
        self.linegraph.x[:, -self.num_charger_types:] = new_charger_config

        for idx, action in enumerate(actions):
            if action == 0:
                continue
            charger = self.charger_list[action]
            link_id = self.edge_mapping.inv[idx]
            ET.SubElement(
                chargers,
                "charger",
                id=str(idx),
                link=str(link_id),
                plug_power=str(charger.plug_power),
                plug_count=str(charger.plug_count),
                type=charger.type,
            )

        tree = ET.ElementTree(chargers)
        with open(self.charger_xml_path, "wb") as f:
            f.write(b'<?xml version="1.0" ?>\n')
            f.write(
                b'<!DOCTYPE chargers SYSTEM "http://matsim.org/files/dtd/chargers_v1.dtd">\n'
            )
            tree.write(f)

        curr_charger_config = self.linegraph.x[:, -self.num_charger_types:]
        new_charger_config = torch.zeros_like(curr_charger_config)
        new_charger_config[torch.arange(new_charger_config.shape[0]), actions] = 1
        self.linegraph.x[:, -self.num_charger_types:] = new_charger_config

    def get_charger_cost_reward(self):
        freeway_length_idx = self.edge_attr_mapping['length']
        static_charger_idx = self.edge_attr_mapping['default']
        dynamic_charger_idx = self.edge_attr_mapping['dynamic']
        num_static_chargers = torch.sum(self.linegraph.x[:,static_charger_idx])
        mask = self.linegraph.x[:, dynamic_charger_idx] == 1
        dynamic_charger_idxs = mask.nonzero(as_tuple=False).squeeze(1)
        vals_denormalized = self._min_max_normalize(self.linegraph.x[:, :-self.num_charger_types], reverse=True)

        length_m = vals_denormalized[dynamic_charger_idxs, freeway_length_idx]
        length_km = length_m * 0.001
        total_length_km = torch.sum(length_km)

        static_cost = StaticCharger.price * num_static_chargers
        dynamic_cost = DynamicCharger.price * total_length_km
        total_cost = static_cost + dynamic_cost
        self.charger_cost = total_cost.item()

        return (total_cost / self.max_charger_cost).item()
    
    def create_chargers_from_graph(self):
        """
        Create a chargers XML file for MATSim using a multi-discrete action space.
        """
        chargers = ET.Element("chargers")
    
        charger_config = self.linegraph.x[:,-self.num_charger_types:]

        for link_idx, config in enumerate(charger_config):
            charger = self.charger_list[torch.nonzero(config, as_tuple=False)[0]]
            if charger.type == "none":
                continue
            
            link_id = self.edge_mapping.inv[link_idx]
            ET.SubElement(
                chargers,
                "charger",
                id=str(link_idx),
                link=str(link_id),
                plug_power=str(charger.plug_power),
                plug_count=str(charger.plug_count),
                type=charger.type,
            )

        tree = ET.ElementTree(chargers)
        with open(self.charger_xml_path, "wb") as f:
            f.write(b'<?xml version="1.0" ?>\n')
            f.write(
                b'<!DOCTYPE chargers SYSTEM "http://matsim.org/files/dtd/chargers_v1.dtd">\n'
            )
            tree.write(f)

    
    def create_chargers_xml_gymnasium(
        self, 
        actions: spaces.MultiDiscrete
    ):
        """
        Create a chargers XML file for MATSim using a multi-discrete action space.
        """
        chargers = ET.Element("chargers")

        for cluster_idx, cluster_action in enumerate(actions):
            for action_idx, action in enumerate(cluster_action[self.available_actions[cluster_idx]]):
                if action == 0:
                    continue
                charger = self.charger_list[action]
                link_idx = self.clusters[cluster_idx][action_idx]
                link_id = self.edge_mapping.inv[link_idx]
                ET.SubElement(
                    chargers,
                    "charger",
                    id=str(link_idx),
                    link=str(link_id),
                    plug_power=str(charger.plug_power),
                    plug_count=str(charger.plug_count),
                    type=charger.type,
                )

        tree = ET.ElementTree(chargers)
        with open(self.charger_xml_path, "wb") as f:
            f.write(b'<?xml version="1.0" ?>\n')
            f.write(
                b'<!DOCTYPE chargers SYSTEM "http://matsim.org/files/dtd/chargers_v1.dtd">\n'
            )
            tree.write(f)

    def update_graph_attr(self, actions):
        pass

    def get_graph(self):
        """
        Returns the graph representation of the MATSim network.

        Returns:
            Data: The graph representation.
        """
        return self.graph
