import xml.etree.ElementTree as ET
import torch
import shutil
from torch_geometric.data import Dataset
from torch_geometric.transforms import LineGraph
from gymnasium import spaces
from torch_geometric.data import Data
from pathlib import Path
from bidict import bidict
from harl.envs.ocp.chargers import Charger, NoneCharger, StaticCharger, DynamicCharger
from sklearn.cluster import KMeans
import numpy as np
import os
from datetime import datetime
from copy import deepcopy as dc


class MatsimXMLDataset(Dataset):
    """
    A dataset class for parsing MATSim XML files and creating a graph
    representation using PyTorch Geometric.
    """

    def __init__(
        self,
        config_path: Path,
        num_clusters: int,
    ):
        """
        Initializes the MatsimXMLDataset.

        Args:
            config_path (Path): Path to the MATSim configuration file.
            time_string (str): Unique identifier for temporary directories.
        """
        super().__init__(transform=None)

        time_string = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.setup_dirs(config_path, time_string)
        
        self.num_clusters = num_clusters
        self.charger_cost = 0


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
        self.parse_charger_network_get_charger_cost()

    def setup_dirs(self, config_path, time_string):
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

    def create_edge_attr_mapping(self):
        """
        Creates a mapping of edge attributes to their indices.
        """
        self.edge_attr_mapping = {"length": 0, "freespeed": 1, "capacity": 2, "slopes":3}
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
                torch.min(self.graph.edge_attr[:, :3], dim=0).values,
                torch.max(self.graph.edge_attr[:, :3], dim=0).values,
            ]
        )

        self.graph.edge_attr[:, :3] = self._min_max_normalize(
            self.graph.edge_attr[:, :3]
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

    def parse_charger_network_get_charger_cost(self):
        """
        Parses the charger network XML file and calculates the total charger
        cost.

        Returns:
            float: Total cost of chargers in the network.
        """
        cost = 0
        tree = ET.parse(self.charger_xml_path)
        root = tree.getroot()

        # Reset the values of the charger placements
        self.graph.edge_attr[:, 3:] = torch.zeros(
            self.graph.edge_attr.shape[0], self.graph.edge_attr[:, 3:].shape[1]
        )

        for charger in root.findall(".//charger"):
            link_id = charger.get("link")
            charger_type = charger.get("type")
            if charger_type is None:
                charger_type = StaticCharger.type

            if charger_type == StaticCharger.type:
                cost += StaticCharger.price
            elif charger_type == DynamicCharger.type:
                link_idx = self.edge_mapping[link_id]
                link_attr = self.graph.edge_attr[link_idx]
                link_attr_denormalized = self._min_max_normalize(
                    link_attr[:4], reverse=True
                )
                link_len_km = (
                    link_attr_denormalized[self.edge_attr_mapping["length"]] * 0.001
                )
                cost += DynamicCharger.price * link_len_km

            self.graph.edge_attr[self.edge_mapping[link_id]][
                self.edge_attr_mapping[charger_type]
            ] = 1

        # Update the rest of the links to have no charger
        tree = ET.parse(self.network_xml_path)
        root = tree.getroot()
        for link in root.findall(".//link"):
            link_id = link.get("id")

            if not (
                self.graph.edge_attr[self.edge_mapping[link_id]][
                    self.edge_attr_mapping["default"]
                ]
                == 1
                or self.graph.edge_attr[self.edge_mapping[link_id]][
                    self.edge_attr_mapping["dynamic"]
                ]
                == 1
            ):
                self.graph.edge_attr[self.edge_mapping[link_id]][
                    self.edge_attr_mapping["none"]
                ] = 1

        self.charger_cost = cost
        return cost
    
    def create_chargers_xml_gymnasium(
        self, 
        actions: spaces.MultiDiscrete
    ):
        """
        Create a chargers XML file for MATSim using a multi-discrete action space.
        """
        chargers = ET.Element("chargers")

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

    def get_graph(self):
        """
        Returns the graph representation of the MATSim network.

        Returns:
            Data: The graph representation.
        """
        return self.graph
