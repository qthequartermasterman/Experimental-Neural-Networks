import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['plain_connected_net']  # The networks contained in this file.


def convert_dna_to_list(dna_string, num_classes):
    # dna_string is a list of numbers separated by spaces
    # num_classes is the number of cifar classes
    layer_counts_as_strings = dna_string.split(' ')
    layer_counts_as_ints = [int(count) for count in layer_counts_as_strings]
    layer_counts_as_ints.append(num_classes)
    return layer_counts_as_ints


class FullyConnectedNet(nn.Module):
    def __init__(self, layer_data):
        # layer_data is a list of the number of nodes in each successive layer
        super(FullyConnectedNet, self).__init__()
        self.image_size = 28
        self.layers = [
            nn.Linear(layer_data[idx - 1], current_layer) if idx > 0 else nn.Linear(self.image_size ** 2, current_layer)
            for idx, current_layer in enumerate(layer_data)]
        self.sequence = nn.Sequential(*self.layers)

    def forward(self, x):
        y = x.view(x.size(0), -1)
        y = self.sequence(y)
        return y


def plain_connected_net(dna_string, num_classes=1000):
    layer_list = convert_dna_to_list(dna_string, num_classes)
    return FullyConnectedNet(layer_list)
