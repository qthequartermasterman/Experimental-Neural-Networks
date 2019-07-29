# noinspection SpellCheckingInspection
"""Pre-activation ResNet in PyTorch.

SResNet50 created by Xu Ma (https://github.com/13952522076/DNN)

Modified 15 July 2019 by Andrew Sansom
Added some experimental modifications to SResNet50 to attempt to improve accuracy
Skip connections in the attention block (SResNet50WithSkips) and Dilation in the convolutional modules in the attention
blocks (SResNet50WithDilation) both improve accuracy of a ResNet50. For more complete results, see the /records/cifar100
directory.

Modified 18 July 2019 by Andrew Sansom
After the results of the skip connections and dilations, we combined the two (as well as implementing a 'squishmoid'
activation function, as per Mara McGuire), but these combined results were lackluster, with no noticeable improvement
over just using dilation.


Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import math
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

__all__ = ['SResNet50', 'SResNet50WithSkips', 'SResNet18WithSkips', 'SResNet50WithDilation', 'SResNet18', 'SlimSResNet',
           'SResNet50WithSkipsDilationAndSquishmoid']


class Squishmoid(nn.Module):
    def forward(self, x):
        squish_factor = 5
        return nn.Sigmoid()(squish_factor * x)


class ChannelAvgPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


class SpatialLayer(nn.Module):
    def __init__(self):
        super(SpatialLayer, self).__init__()
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

    def forward(self, x):
        y = self.gather(x)
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.excite(y)
        return x * y


class SpatialLayerWithSkips(nn.Module):
    def __init__(self):
        super(SpatialLayerWithSkips, self).__init__()
        self.channelavgpool = ChannelAvgPool()
        self.conv1 = nn.Conv2d(1, 1, 5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(3, 1, 1, bias=False)

    def forward(self, x):
        # Gather
        y = self.channelavgpool(x)
        pooling_skip = y
        # print(pooling_skip.shape)
        y = self.conv1(y)
        y = self.relu(y)
        conv1_skip = y
        # print(conv1_skip.shape)
        y = self.conv2(y)
        y = self.relu(y)
        # print(y.shape)

        # Excite
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')  # upsample
        # y = y + pooling_skip + conv1_skip  # All of these layers are the same size
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = torch.cat([y, pooling_skip, conv1_skip], dim=1)
        # print(y.shape)
        y = self.conv4(y)
        # print(y.shape)
        y = nn.Sigmoid()(y)
        # y is now some "mask" that will tell us how relevant each pixel "should" be.

        return x * y

        # globalskip = x  # Global Residual Connection
        # y = self.gather(x)
        # y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        # y = self.excite(y)
        # return x*y + globalskip


class SpatialLayerWithDilation(nn.Module):
    def __init__(self):
        super(SpatialLayerWithDilation, self).__init__()
        self.gather = nn.Sequential(
            ChannelAvgPool(),
            nn.Conv2d(1, 1, 5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, stride=2, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.excite = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=2, dilation=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.gather(x)
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
        y = self.excite(y)
        return x * y


class SpatialLayerWithSkipsDilationAndSquishmoid(nn.Module):
    def __init__(self):
        super(SpatialLayerWithSkipsDilationAndSquishmoid, self).__init__()
        self.channelavgpool = ChannelAvgPool()
        self.conv1 = nn.Conv2d(1, 1, 5, padding=4, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(1, 1, 3, stride=1, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(3, 1, 1, bias=False)

    def forward(self, x):
        # Gather
        y = self.channelavgpool(x)
        pooling_skip = y
        # print(pooling_skip.shape)
        y = self.conv1(y)
        y = self.relu(y)
        conv1_skip = y
        # print(conv1_skip.shape)
        y = self.conv2(y)
        y = self.relu(y)
        # print(y.shape)

        # Excite
        y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')  # upsample
        # y = y + pooling_skip + conv1_skip  # All of these layers are the same size
        # print(y.shape)
        y = self.conv3(y)
        # print(y.shape)
        y = torch.cat([y, pooling_skip, conv1_skip], dim=1)
        # print(y.shape)
        y = self.conv4(y)
        # print(y.shape)
        y = Squishmoid()(y)
        # y is now some "mask" that will tell us how relevant each pixel "should" be.

        return x * y


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = SpatialLayer()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class PreActBlockWithSkips(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockWithSkips, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = SpatialLayerWithSkips()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.se = SpatialLayer()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class PreActBottleneckWithDilation(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckWithDilation, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.se = SpatialLayerWithDilation()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class PreActBottleneckWithSkips(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckWithSkips, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.se = SpatialLayerWithSkips()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class PreActBottleneckWithSkipsDilationAndSquishmoid(nn.Module):
    """Pre-activation version of the original Bottleneck module."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckWithSkipsDilationAndSquishmoid, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.se = SpatialLayerWithSkipsDilationAndSquishmoid()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, init_weights=True, num_input_channels=3):
        # num_input_channels might be more than 3 if the image is RGBA (i.e. has a transparency mask)
        print('Inside resnet. Num input channels', num_input_channels)
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m, 'bias.data'):
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# noinspection PyPep8Naming
def SlimSResNet(num_classes=1000, input_channels=3):
    return PreActResNet(PreActBlock, [1, 1, 1, 1], num_classes, num_input_channels=input_channels)


# noinspection PyPep8Naming
def SResNet18(num_classes=1000, input_channels=3):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes, num_input_channels=input_channels)


# noinspection PyPep8Naming
def SResNet34(num_classes=1000):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes)


# noinspection PyPep8Naming
def SResNet50(num_classes=1000, input_channels=3):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes, num_input_channels=input_channels)


# noinspection PyPep8Naming
def SResNet18WithSkips(num_classes=1000):
    return PreActResNet(PreActBlockWithSkips, [2, 2, 2, 2], num_classes)


# noinspection PyPep8Naming
def SResNet50WithSkips(num_classes=1000):
    return PreActResNet(PreActBottleneckWithSkips, [3, 4, 6, 3], num_classes)


# noinspection PyPep8Naming
def SResNet50WithDilation(num_classes=1000):
    return PreActResNet(PreActBottleneckWithDilation, [3, 4, 6, 3], num_classes)


# noinspection PyPep8Naming
def SResNet50WithSkipsDilationAndSquishmoid(num_classes=1000):
    return PreActResNet(PreActBottleneckWithSkipsDilationAndSquishmoid, [3, 4, 6, 3], num_classes)


# noinspection PyPep8Naming
def SResNet101(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes)


# noinspection PyPep8Naming
def SResNet152(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes)


# TESTING
def test():
    net = SResNet50(num_classes=100)
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())


def test_skips():
    net = SResNet50WithSkips(num_classes=100)
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())


def test_dilation():
    net = SResNet50WithDilation(num_classes=100)
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())


# test()
def convsize(padding, dilation, kernel, stride, input_size):
    return math.floor((input_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
