"""
Attention Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CompoundAttentionNet']  # The networks contained in this file.


# Modules that we will utilize

class Identity(nn.Module):
    # We will replace some SELayers with Identity Blocks
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ChannelAvgPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


class SEBlock(nn.Module):
    def __init__(self, channel):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        post = x * y.expand_as(x)
        if residual is not None:
            return nn.Conv2d(post.shape[1] + residual.shape[1], x.shape[1], kernel_size=1)(
                torch.cat([post, residual], dim=1))
            # return (post+residual)/2  # add the skip connection from previous layer
        else:
            return post


class SpatialBlock(nn.Module):
    def __init__(self):
        super(SpatialBlock, self).__init__()
        self.gather = nn.Sequential(
            ChannelAvgPool(),
            nn.Conv2d(1, 1, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.excite = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        y = self.gather(x)
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.excite(y)
        if residual is not None:
            return nn.Conv2d(x.shape[1] + residual.shape[1], x.shape[1], kernel_size=1)(
                torch.cat([x * y, residual], dim=1))
            # return (x * y + residual)/2  # add the skip connection from previous layer
        else:
            return x * y


class ConvolutionBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ConvolutionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class Branch(nn.Module):
    def __init__(self, left_branch, right_branch):
        # left_branch and right_branch must have the same output size, otherwise pytorch will throw errors.
        super(Branch, self).__init__()
        self.left = left_branch
        self.right = right_branch

    def forward(self, x):
        # Defaults to simply returning an element-wise sum of the outputs of the two branches.
        return self.left(x) + self.right(x)


class Layer(nn.Module):
    def __init__(self, planes, stride=1):
        super(Layer, self).__init__()
        self.convolution_block = ConvolutionBlock(planes, planes, stride)
        self.se_block = SEBlock(planes)
        self.spatial_block = SpatialBlock()

    def forward(self, x, se_residual=None, spatial_residual=None):
        conv = self.convolution_block(x)
        se = self.se_block(x, se_residual)
        spatial = self.spatial_block(x, spatial_residual)
        post_se = conv * se
        post_spatial = post_se * spatial
        post_spatial = nn.functional.relu(post_spatial)
        return post_spatial, se, spatial


class FullNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(FullNetwork, self).__init__()
        self.internal_size = 16
        self.conv1 = nn.Conv2d(3, self.internal_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = Layer(self.internal_size)
        # self.layer2 = Layer(64)
        # self.layer3 = Layer(64)
        self.layer4 = Layer(self.internal_size)
        self.linear = nn.Linear(self.internal_size * 32 * 32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out, se, spatial = self.layer1(out, None, None)  # There are no skip connections yet
        # out, se, spatial = self.layer2(out, se, spatial)  # Add skip connection
        # out, se, spatial = self.layer3(out, se, spatial)  # Add skip connection
        out, _, _ = self.layer4(out, se, spatial)  # Add skip connection
        out = out.view(out.size(0), -1)
        out = self.linear(out)  # Classify
        return out


def test():
    random_input = torch.rand((1, 3, 32, 32))
    net = FullNetwork()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Parameters: {}'.format(count_parameters(net)))
    return net(random_input)


def CompoundAttentionNet(num_classes=100):
    return FullNetwork(num_classes)
