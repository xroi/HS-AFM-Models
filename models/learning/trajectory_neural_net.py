import torch
import torch.nn as nn


class TrajectoryNeuralNet(nn.Module):
    def __init__(self):
        super(TrajectoryNeuralNet, self).__init__()
        self.l1 = nn.Linear(5, 25, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(25, 2, dtype=torch.float64)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
