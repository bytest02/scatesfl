import os

from torchvision import datasets, transforms

import numpy as np
from collections import defaultdict

def get_datasets(dataset_name):
    dataset_dir = os.path.join('/data', 'Datasets', dataset_name)
    if dataset_name == 'CIFAR10':
        dataset_mean, dataset_std = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
    else: # 'MNIST'
        dataset_mean, dataset_std = (0.1307,), (0.3081,)

    train_transforms = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(10),
                           transforms.Pad(3),
                           transforms.CenterCrop(32),
                           transforms.ToTensor(),
                           transforms.Normalize(dataset_mean, dataset_std)
                       ])

    test_transforms = transforms.Compose([
                           transforms.Pad(3),
                           transforms.CenterCrop(32),
                           transforms.ToTensor(),
                           transforms.Normalize(dataset_mean, dataset_std)
                       ])

    if dataset_name == 'CIFAR10':
        dataset_train = datasets.CIFAR10(root = dataset_dir, train = True, download = True, transform = train_transforms)
        dataset_test = datasets.CIFAR10(root = dataset_dir, train = False, download = True, transform = test_transforms)
    else:
        dataset_train = datasets.MNIST(root = dataset_dir, train = True, download = True, transform = train_transforms)
        dataset_test = datasets.MNIST(root = dataset_dir, train = False, download = True, transform = test_transforms)
    return dataset_train, dataset_test

# iid_distribute_dataset_to_users() will create an IID datasets based on number of users
def iid_distribute_dataset_to_users(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    # return ths dictionary of the indices -- each key corresponding to a client

def non_iid_distribute_dataset_to_users(dataset, num_users, shard_size=200):
    # Create a dictionary to hold image index by class
    class_dict = defaultdict(list)
    # Sort image index by class
    for index, label in enumerate(dataset.targets):
        class_dict[label].append(index)
    # Now slice each class into shards
    sliced_datasets = {}
    for class_label, indice in class_dict.items():
        sliced_datasets[class_label] = [indice[i:i+shard_size] for i in range(0, len(indice), shard_size)]
    # create a dictionary to hold the image indices for each user
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = []
    # Gaussian distribution parameters
    mean = num_users / 2  # Center of the distribution
    std_dev = num_users / 4  # Spread of the distribution
    for class_label, slices in sliced_datasets.items():
        for shard in slices:
            # Sample an user index using Gaussian sampling
            sampled_user = int(np.clip(np.random.normal(mean, std_dev), 0, num_users - 1))
            dict_users[sampled_user] = dict_users[sampled_user] + shard
    return dict_users

