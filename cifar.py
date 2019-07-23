"""
Train a neural network on CIFAR with PyTorch. Adapted from Xu Ma's CIFAR script(https://github.com/13952522076/DNN).
Added cross-platform support (including support for devices without a GPU).
Added the ability to preprocess images, giving them an alpha channel that corresponds to how likely a pixel is to be
relevant. For details see /models/imagegradients.py.

Call this script from the command line with its proper arguments. This script should work properly on any environment
that has pytorch and torchvision installed.

Commandline Arguments

--lr=0.1            Starting learning rate for training process
--resume=False      Resume training using a pretrained model. Will check for a file with the name of the network
                    in /checkpoint directory
--netName           This must be a network name from the installed networks in /models.
                    To install a network, see the section below.
--bs=512            Batch size to be used in training
--es=150            Number of epochs to train ('Epoch Size')
--cifar=100         Version of CIFAR to use. Must be 10 or 100.
--fix_seed=123      Seed for the random number generator used by the Stochastic Gradient Descent
--pre_process=False Indicates whether the pre-processing alpha channel should be added before feeding it into the network.
                    NOTE: since the images will now have 4 color channels, this breaks many neural networks.

e.g.
    python3 cifar.py --netName=PreActResNet18 --cifar=10 --bs=512


HOW TO USE THIS SCRIPT WITH A CUSTOM NEURAL NETWORK
1. Create a python file in /models with a function that returns an instance of your network.
2. Override the __all__ variable to be a list that includes a string containing the name of the above function.
e.g. __all__ = ['SResNet50', 'SResNet50WithSkips', 'SResNet18WithSkips', 'SResNet50WithDilation']
3. Make sure that /models/__init__.py includes the line:
    from .FileNameHere import *
4. Run this script from a command line interface with the proper arguments.
"""
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim

import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import random

import models as models
from utils import *

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))


# print(model_names)

# str2bool will us to accept more than just 'True' or 'False' from the command line, since not santizing the input
# breaks this script everytime.
def str2bool(v):
    # v should be some string.
    return v.lower() in ("yes", "true", "t", "1")


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--netName', default='PreActResNet18', choices=model_names, type=str, help='choosing network')
parser.add_argument('--bs', default=512, type=int, help='batch size')
parser.add_argument('--es', default=150, type=int, help='epoch size')
parser.add_argument('--cifar', default=100, type=int, help='dataset classes number')
parser.add_argument('--fix_seed', default=123, help='Fix random seed')
parser.add_argument('--pre_process', default=False, type=str2bool, help='apply an image mask before training')
args = parser.parse_args()

# This is encased inside a main function, because Windows does not handle the parallelization properly without it.
def main():
    global best_acc
    print('preprocess', args.pre_process)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.fix_seed > 0:
        # Seed model
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        cudnn.deterministic = True
        print("SEED MODEL: Fix seed as ", args.fix_seed)
    else:
        print("SEED MODEL: Using random seed.")

    # Data Preparation. This will download the dataset if necessary.
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.cifar == 100:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    else:
        args.cifar = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=1)

    if args.cifar == 100:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=1)

    # Call/Configure the Model
    print('==> Building model..')
    number_of_input_channels = 4 if args.pre_process else 3
    print('number of input channels', number_of_input_channels)
    try:
        net = models.__dict__[args.netName](num_classes=args.cifar, input_channels=number_of_input_channels)
    except:
        net = models.__dict__[args.netName]()

    para_numbers = count_parameters(net)
    print("Total parameters number is: " + str(para_numbers))

    net = net.to(device)

    # If GPUs are available, then let's utilize them!
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # If resume is True in the arguments, then we will load the previous saved model.
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_path = './checkpoint/ckpt_cifar_' + str(args.cifar) + '_' + args.netName + str(
            args.pre_process) + '.t7'
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        print("BEST_ACCURACY: " + str(best_acc))
        start_epoch = checkpoint['epoch']

    # Training
    # Create the directory for the results if it doesn't exist yet.
    try:
        os.makedirs("./records/cifar100")
    except FileExistsError:
        # directory already exists
        pass
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    def train(epoch_number):
        adjust_learning_rate(optimizer, epoch_number, args.lr)  # This is in the utils file.
        print('\nEpoch: %d   Learning rate: %f' % (epoch_number, optimizer.param_groups[0]['lr']))
        print("\nAllocated GPU memory:",
              torch.cuda.memory_allocated() if torch.cuda.is_available() else 'CUDA NOT AVAILABLE')
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs_masked = models.imagegradients.mask_pytorch_image_batch(
                inputs).to(device) if args.pre_process else None  # Check if we should mask the images.
            inputs, targets = inputs.to(device), targets.to(device)  # Move the stuff to GPU after CPU work
            # print('input same:', inputs-inputs2)
            optimizer.zero_grad()
            outputs = net(inputs_masked) if args.pre_process else net(inputs)  # Check if we should mask the images.
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        file_path = './records/cifar100/cifar_' + str(
            args.cifar) + '_' + args.netName + str(args.pre_process) + '_train.txt'
        record_str = str(epoch_number) + '\t' + "%.3f" % (train_loss / (batch_idx + 1)) + '\t' + "%.3f" % (
                100. * correct / total) + '\n'
        write_record(file_path, record_str)

    def test(epoch_number):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs_masked = models.imagegradients.mask_pytorch_image_batch(
                    inputs).to(device) if args.pre_process else None  # Check if we should mask the images.
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs_masked) if args.pre_process else net(inputs)  # Check if we should mask the images
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        file_path = './records/cifar100/cifar_' + str(args.cifar) + '_' + args.netName + str(
            args.pre_process) + '_test.txt'
        record_str = str(epoch_number) + '\t' + "%.3f" % (test_loss / (batch_idx + 1)) + '\t' + "%.3f" % (
                100. * correct / total) + '\n'
        write_record(file_path, record_str)

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch_number,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_path = './checkpoint/ckpt_cifar_' + str(args.cifar) + '_' + args.netName + str(
                args.pre_process) + '.t7'
            torch.save(state, save_path)
            best_acc = acc

    for epoch in range(start_epoch, start_epoch + args.es):
        train(epoch)
        test(epoch)

    # write statistics to files

    statis_path = './records/cifar100/STATS_' + args.netName + str(args.pre_process) + '.txt'
    if not os.path.exists(statis_path):
        # os.makedirs(statis_path)
        os.system(r"touch {}".format(statis_path))
    f = open(statis_path, 'w+')
    statis_str = "============\nDevices:" + device + "\n"
    statis_str += '\n===========\nargs:\n'
    statis_str += args.__str__()
    statis_str += '\n==================\n'
    statis_str += "BEST_accuracy: " + str(best_acc)
    statis_str += '\n==================\n'
    statis_str += "Total parameters: " + str(para_numbers)
    f.write(statis_str)
    f.close()


# Wrapped in a main function to preserve functionality on Windows.
if __name__ == '__main__':
    best_acc = 0  # best test accuracy
    main()
