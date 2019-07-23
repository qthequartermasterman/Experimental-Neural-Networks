import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """A 3x3 Convolution layer with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """A 1x1 Convolution layer"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return torch.sqrt(2+2*((x+1) ** 2)) - 2


class IdentityBlock(nn.Module):
    def __init__(self, inplanes, f, filters, stride=1):
        super(IdentityBlock, self).__init__()
        self.f = f

        self.F1, self.F2, self.F3 = filters

        # layer definitions
        self.conv1 = conv1x1(inplanes, self.F1)
        self.bn1 = nn.BatchNorm2d(self.F1)
        self.relu = CustomActivation()
        self.conv2 = conv3x3(self.F1, self.F2)
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.conv3 = conv1x1(self.F2, self.F3)
        self.bn3 = nn.BatchNorm2d(self.F3)

    def forward(self, x):
        x_shortcut = x  # this will become our skip connection

        # stage1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # stage2
        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        # stage3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # Add the skip connection
        out = out + x_shortcut
        out = self.relu(out)
        return out

class ConvolutionalBlock(nn.Module):
    def __init__(self, inplanes, f, filters, stride=1):
        super(ConvolutionalBlock, self).__init__()
        self.f = f
        self.F1, self.F2, self.F3 = filters

        # layer definitions
        self.conv1 = conv1x1(inplanes, self.F1, stride=stride)
        self.bn1 = nn.BatchNorm2d(self.F1)
        self.relu = CustomActivation()
        self.conv2 = conv3x3(self.F1, self.F2)
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.conv3 = conv1x1(self.F2, self.F3)
        self.bn3 = nn.BatchNorm2d(self.F3)
        self.skipconv = conv1x1(inplanes, self.F3, stride=stride)
        self.skipbn = nn.BatchNorm2d(self.F3)

    def forward(self, x):
        x_shortcut = x  # this will become our skip connection

        # stage1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # stage2
        #out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        # stage3
        out = self.conv3(out)
        out = self.bn3(out)

        # shortcut path
        x_shortcut = self.skipconv(x_shortcut)
        x_shortcut = self.skipbn(x_shortcut)

        # Add the skip connection
        out = out + x_shortcut
        out = self.relu(out)
        return out

class Res_Net_50(nn.Module):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    """

    def __init__(self, num_classes=1000):
        super(Res_Net_50, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 1, [64, 64, 256])
        self.layer2 = self._make_layer(256, 2, [128, 128, 512])
        self.layer3 = self._make_layer(512, 3, [256, 256, 1024], stride=2)
        self.layer4 = self._make_layer(1024, 4, [256, 256, 2048], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(2048, num_classes)

    def _make_layer(self, inplanes, layer_number, filters, stride=1):
        layer_count = {1: 2, 2: 3, 3: 2, 4: 2}  # the key is the layer_number. The value is how many identity blocks
        # The original paper has 3: 5. We simplified it for computation sake
        layers = []
        layers.append(ConvolutionalBlock(inplanes, f=3, filters=filters))
        for _ in range(2, layer_count[layer_number]):
            layers.append(IdentityBlock(inplanes*2, f=3, filters=filters))

        return nn.Sequential(*layers)

    def forward(self, x):
        # stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        # stage 2 - 5
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Stage 6 (Avg. Pooling and fully connected)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)  # Flatten it
        x = self.fully_connected(x)

        return x


net = Res_Net_50(num_classes=10)
print(net)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/home/CIFAR-10 Classifier Using CNN in PyTorch/data/',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=8,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range
    image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1,2,0)

dataiter = iter(trainloader)
images, labels = dataiter.next()

fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
for idx, image in enumerate(images):
    axes[idx].imshow(convert_to_imshow_format(image))
    axes[idx].set_title(classes[labels[idx]])
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

learning_rate = .05
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

import os

model_directory_path = '/home/CIFAR-10 Classifier Using CNN in PyTorch/model/'
model_path = model_directory_path + 'cifar-10-cnn-model-custom.pt'

if not os.path.exists(model_directory_path):
    os.makedirs(model_directory_path)
if os.path.isfile(model_path):
    # loading the trained parameters
    net.load_state_dict(torch.load(model_path))
    print('Loaded model parameters from disk')
else:
    for epoch in range(100):
        graph_x, errors, losses = [], [], []
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            total_correct = 0
            # get inputs
            inputs, labels = data
            # print("labels:" + str(labels))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward, backward, then optimize
            outputs = net(inputs)
            # print("outputs: " + str(outputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            graph_x.append(i)
            error_for_batch = 1 - total_correct / len(labels)
            errors.append(error_for_batch)
            losses.append(running_loss)
            if i % 100 == 0:
                # show the graph every 50 minibatches
                plt.plot(graph_x, errors)
                plt.draw()
                plt.pause(0.001)
            if i % 1 == 0:
                # Every 1 minibatches, we print
                print('[%d, %5d] loss: %.3f error: %.2f%%' % (epoch+1, i+1, running_loss, error_for_batch*100))
                running_loss = 0.0
            if i % 5000 == 0:
                learning_rate = learning_rate / 10
                optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        torch.save(net.state_dict(), model_path)
        print('Saved model parameters to disk.')
        with open('./losses.txt', 'a+') as file:
            for i in losses:
                file.write(str(i) + '\n')
        with open('./errors.txt', 'a+') as file:
            for i in errors:
                file.write(str(i) + '\n')
    print('Finished training.')


dataiter = iter(testloader)
images, labels = dataiter.next()

fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
for idx, image in enumerate(images):
    axes[idx].imshow(convert_to_imshow_format(image))
    axes[idx].set_title(classes[labels[idx]])
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

outputs = net(images)

sm = nn.Softmax(dim=1)
sm_outputs = sm(outputs)
print(sm_outputs)
probs, index = torch.max(sm_outputs, dim=1)

for p, i in zip(probs, index):
    print('{0} - {1:.4f}'.format(classes[i], p))

total_correct = 0
total_images = 0
confusion_matrix = np.zeros([10,10], int)
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_images += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        for i, l in enumerate(labels):
            confusion_matrix[l.item(), predicted[i].item()] += 1

model_accuracy = total_correct / total_images * 100
print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))
