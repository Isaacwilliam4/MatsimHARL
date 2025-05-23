import torch
import torch.nn as nn


class MatsimMLP(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.linear1 = nn.Linear(input_channels, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 16)
        self.fc = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Pass data through GCN layers"""
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.fc(x)
        return torch.mean(x)
