import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_valid_layer_size(int_input):
    if int_input > 0:
        return int_input
    else:
        return 1


def convert_dna_to_list(dna_string):
    # dna_string is a list of numbers separated by spaces
    # num_classes is the number of cifar classes
    layer_counts_as_strings = dna_string.split(' ')
    layer_counts_as_ints = [int(count) for count in layer_counts_as_strings]
    return layer_counts_as_ints


def add_layer(input_string):
    layer_list = convert_dna_to_list(input_string)  # Obtain a list of all of the layer_sizes
    num_layers = len(layer_list)  # We will use this to choose a random place to insert a layer
    random_place = np.random.randint(0, num_layers)
    mean_layer_size, stddev_layer_size = np.mean(layer_list), np.std(layer_list, ddof=1)
    random_layer_size = np.random.normal(mean_layer_size, stddev_layer_size)
    random_layer_size_int = make_valid_layer_size(int(round(random_layer_size)))
    layer_list.insert(random_place, random_layer_size_int)
    return ' '.join((str(x) for x in layer_list))


def resize_layer(input_string):
    layer_list = convert_dna_to_list(input_string)
    num_layers = len(layer_list)  # We will use this to choose a random place to insert a layer
    random_place = np.random.randint(0, num_layers)
    random_change_in_size = int(round(np.random.normal(0, 2.5)))  # We don't want huge changes in layer sizes.
    layer_list[random_place] += random_change_in_size
    layer_list[random_place] = make_valid_layer_size(layer_list[random_place])
    return ' '.join((str(x) for x in layer_list))


def remove_layer(input_string):
    layer_list = convert_dna_to_list(input_string)  # Obtain a list of all of the layer_sizes
    num_layers = len(layer_list)  # We will use this to choose a random place to insert a layer
    random_place = np.random.randint(0, num_layers)
    layer_list.pop(random_place)
    return ' '.join((str(x) for x in layer_list))


def test():
    dna = '2 3 5 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101 103'
    for idx in range(0, 100000):
        dna = add_layer(dna)
        print(idx, dna)
        dna = resize_layer(dna)
        print(idx, dna)
        dna = remove_layer(dna)
        print(idx, dna)
