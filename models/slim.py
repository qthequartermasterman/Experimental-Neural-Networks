"""
Attention Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SlimNet']  # The networks contained in this file.


class SlimNetwork(nn.Module):
    def __init__(self, num_channels=3):
        super(SlimNetwork, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * num_channels, 100)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        return self.fc1(out)


def SlimNet(num_classes=100, input_channels=3):
    return SlimNetwork(input_channels)
