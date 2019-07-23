import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['ShallowWideNet7']  # The networks contained in this file.


def calculate_padding(kernel_size, stride, dilation, output_size):
    # calculates the padding needed for a Conv2d module to have the same input and output size
    # This formula comes from the docs for nn.Conv2D, then solved for the padding, with input_size=output_size
    padding = (dilation * (kernel_size - 1) + (output_size - 1) * (stride - 1)) / 2
    return int(math.ceil(padding))


def auto_conv2d(input_channels, output_channels, kernel_size, stride, dilation, image_size):
    # returns a Conv2d with the identical input and output sizes (though channel count might differ)
    return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                     padding=calculate_padding(kernel_size, stride, dilation, image_size))


class ShallowWideNet(nn.Module):
    def __init__(self, largest_filter, num_classes=100, input_channels=3, inside_channels=32, stride=2, dilation=1):
        super(ShallowWideNet, self).__init__()
        self.image_size = 32  # the size of square input images
        self.largest_filter = largest_filter
        self.inside_channels = inside_channels
        self.conv1 = auto_conv2d(input_channels, inside_channels, kernel_size=1, stride=stride, dilation=dilation,
                                 image_size=self.image_size)
        self.conv3 = auto_conv2d(input_channels, inside_channels, kernel_size=3, stride=stride, dilation=dilation,
                                 image_size=self.image_size)
        self.conv5 = auto_conv2d(input_channels, inside_channels, kernel_size=5, stride=stride, dilation=dilation,
                                 image_size=self.image_size)
        self.conv7 = auto_conv2d(input_channels, inside_channels, kernel_size=7, stride=stride, dilation=dilation,
                                 image_size=self.image_size)
        self.number_of_branches = 4  # This is 1 + number of conv blocks
        # (Currently hard coded until I can find a better way to parallelize them)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear((inside_channels * self.number_of_branches) * (self.image_size ** 2), num_classes)
        # self.linear = nn.Linear(134144, num_classes)  # I don't know why the above equation doesn't generate the proper number of channels.

    def forward(self, x):
        shortcut = x  # shortcut block
        output = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        output = self.relu(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output


def ShallowWideNet7(num_classes=100):
    return ShallowWideNet(7, num_classes=num_classes, input_channels=3)