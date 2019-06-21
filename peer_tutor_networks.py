"""
Peer Tutor Neural Networks
An experimental idea to (potentially) increase the accuracy of neural networks in training

Imagine a classroom where a teacher provides some common instruction for all students, assigns individual homework, and
then after a quiz, has the student that scored highest on each topic tutor the others on that topic.

A Peer Tutor Neural Network follows a similar philosophy, training a set of identical neural networks on separate data
after common instruction, and for every module in the network, adjust the weights of each network to resemble those of
the network that performed best on that module.

After each epoch of training, the program iterates over each layer in the network architecture and calculates the mean
gradient of each layer (i.e. how much that layer has to change to better resemble the data) during backprop. Using those
mean gradients for each layer, it changes the weights of every network's instance of that layer to be the average of
itself and the instance that performed best on the given dataset.

While it was hypothesized that this could allow for faster training, since different networks would take paths to
achieve a local minimum in the loss function, and averaging the weights would more effectively find the minimum, in
practice, this merely resulted in the networks' converging to each other and /then/ converging to a minimum.

This makes sense as the splitting the data up meant that each network was only given approximately 1/n of the data each
epoch (where n is the number of networks).

HOWEVER, compensating by increasing the learning rate /does/ provide tangible benefits over the control (1 network).
TODO: Study how learning rate affects this compensatory effect.
"""

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F

# Basic Neural Network Training constants
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# Set up the peer-classroom environment
number_of_networks = 3
number_of_shared_training_batches = 0  # The number of batches that all networks will be trained on.
number_of_shared_comparison_batches = 20  # The number of training batches that will be reserved for tutoring

# Set up the data to train on.
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

# Define the training/testing functions
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# Record the training losses to file so we can compare the networks later.
def record_training_loss(epoch_number, batch_number, loss, network_id,
                         path='./peer_tutor_model_stats/peer_tutor_training_loss{}.csv'):
    with open(path.format(network_id), 'a') as file:
        file.write('{},{},{}\n'.format(epoch_number, batch_number, loss))  # Write a new line to the csv file.

# Record the testing losses to file so we can compare the networks later.
def record_testing(epoch_number, loss, accuracy, network_id,
                         path='./peer_tutor_model_stats/peer_tutor_testing{}.csv'):
    with open(path.format(network_id), 'a') as file:
        file.write('{},{},{}\n'.format(epoch_number, loss, accuracy))  # Write a new line to the csv file


def train(epoch, network, optimizer, number_of_shared_batches, number_of_networks, number_of_compared_batches):
    print('TRAINING NETWORK WITH ID: {}'.format(network.id))
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx/10 > number_of_compared_batches) and \
                (batch_idx/10 < number_of_shared_batches + number_of_compared_batches
                 or (batch_idx/10 % number_of_networks == network.id)):
            # The first number_of_compared_batches are reserved for the comparison step in training.
            # All networks need to be trained on the shared batches
            # Then we partition the remaining ones with a mod function.
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # Log to the console
            if batch_idx % log_interval == 0:
                print('Train ID: {} Epoch: {} Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    network.id, epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                torch.save(network.state_dict(), './results/model{}.pth'.format(network.id))
                torch.save(optimizer.state_dict(), './results/optimizer{}.pth'.format(network.id))
                record_training_loss(epoch, batch_idx/10, loss.item(), network_id=network.id)
        else:
            if batch_idx % log_interval == 0:
                print('Train ID: {} Epoch: {} Batch: {} -- NO TRAINING DONE'.format(network.id, epoch, batch_idx))
    test(network, epoch=epoch, pre_tutoring=True)  # This is solely for recording the test loss each epoch. Optional.


def test(network, epoch=None, pre_tutoring=False):
    print('TESTING NETWORK WITH ID: {}'.format(network.id))
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    if epoch is not None:
        if pre_tutoring is True:
            # Use a different file path so that it doesn't get mixed in with post tutoring testing stats
            record_testing(epoch, test_loss, 100. * correct / len(test_loader.dataset), network_id=network.id,
                           path='./peer_tutor_model_stats/peer_tutor_testing_before_tutoring{}.csv')
        else:
            record_testing(epoch, test_loss, 100. * correct / len(test_loader.dataset), network_id=network.id)


def test_all_networks(networks, epoch=None):
    for network in networks:
        if epoch is None:
            test(network)
        else:
            test(network, epoch)


def train_all_networks(networks, optimizers, number_of_shared_batches, number_of_networks,
                       number_of_comparison_batches):
    # First, train all of the networks with its allocated data samples.
    # Second, allow the network to tutor themselves
    # Third, test all the networks to see how it improved that epoch
    for epoch in range(1, n_epochs + 1):
        for network, optimizer in zip(networks, optimizers):
            train(epoch, network, optimizer, number_of_shared_batches, number_of_networks,
                  number_of_comparison_batches)
        peer_tutor_all_networks(networks, number_of_networks, number_of_comparison_batches)
        test_all_networks(networks, epoch)


def peer_tutor_all_networks(networks, number_of_networks, number_of_comparison_batches):
    for batch_idx, (data, target) in enumerate(train_loader):
        # We will test each network using the same batch of data.
        if batch_idx/10 < number_of_comparison_batches:
            # We only will test each network with the first few batches that we reserved for comparison
            means = []  # We will append the means of each module in each network's modules into this array
            # Outermost dimension corresponds to individual networks
            # The next dimension is a list of means corresponding to gradients of each module in the network
            for network in networks:
                network.train()
                output = network(data)
                loss = F.nll_loss(output, target)
                loss.backward()  # Calculate the gradients
                network_means = []
                for module in network._modules.items():
                    try:
                        # Calculate the mean gradient of each module in the network
                        mean = module[1].weight.grad.mean().item()
                        network_means.append((module[0], mean))  # We want to keep the name of the module
                    except AttributeError:
                        # network_means.append((module[0], float('NaN')))
                        # I don't think we need to carry around all these NaNs now that we're carrying the module names
                        pass
                    except TypeError:
                        pass
                means.append(network_means)
            # Now that we have all the means in a "matrix", we need to choose the max gradient for each module
            means_transpose = [list(i) for i in zip(*means)]
            # Each row is a network, but we want each row to be the respective modules in each network
            print('Choosing tutors for each layer.')
            winners = [(module[0][0],
                        torch.abs(torch.tensor([individual_module[1] for individual_module in module])).argmax().item())
                       for module in means_transpose]
            # This looks like a mess. This is a list of tuples. The first element in each tuple is the name of a module.
            # The second element in each tuple is the index of the network that has the greatest average mean gradient.
            # From inside to out:
            #   1. create a list of all of the mean gradients via a list comprehension
            #   2. Convert this to a tensor
            #   3. Take the element-wise absolute value of this tensor
            #   4. argmax() returns the index of the maximum value in a 0-dimensional tensor
            #   5. item() extracts the value from the argmax tensor
            # This is not the cleanest way of notating this, but it's the most succinct utilizing list comprehensions.

            # Once we choose the maxes, we then have to determine how to step every other module's parameters toward the
            # winning module's parameters
            for layer in winners:
                layer_name = layer[0]
                winning_network_id = layer[1]
                for network in networks:
                    if network.id is not winning_network_id:
                        print('Adjusting layer {} using the tutor\'s (network {}) weights'.format(layer_name,
                                                                                                  winning_network_id))
                        network._modules[layer_name].weight.data += networks[winning_network_id]._modules[
                            layer_name].weight.data
                        network._modules[layer_name].weight.data /= 2


# Defining the Network
class SingleNetwork(nn.Module):
    def __init__(self, id):
        super(SingleNetwork, self).__init__()
        self.id = id
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Block 2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # Block 3
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# Set up the networks for training
networks = []
optimizers = []
for i in range(0, number_of_networks):
    networks.append(SingleNetwork(i))
    optimizers.append(optim.SGD(networks[i].parameters(), lr=learning_rate, momentum=momentum))


# Test/train each network
train_all_networks(networks, optimizers, number_of_shared_training_batches, number_of_networks,
                   number_of_shared_comparison_batches)
test_all_networks(networks)
