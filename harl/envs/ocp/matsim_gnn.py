import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch, Data

class MatsimGNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv1 = GCNConv(input_channels, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 16)
        self.fc = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """Pass data through GCN layers"""
        edge_index = edge_index.to(torch.int64)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.fc(x)
        return torch.mean(x)
