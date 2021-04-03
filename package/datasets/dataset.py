
import _pickle as cPickle
from scipy import sparse as sp

import os
import json
import pandas as pd
import numpy as np
import scipy.sparse

import torch.utils.data as data


def load_planetoid_data(directory, dataset_name, dataset_style):
    print(f"Loading data from directory {directory}")
    directory_files = os.listdir(directory)
    matching_files = [f for f in directory_files if dataset_name in f and dataset_style in f]
    datasets = {}
    names = ['x', 'y', 'tx', 'ty', 'allx', 'graph']
    for f in matching_files:
        fullpath = os.path.join(directory, f)
        suffix = f.split('.')[-1]
        # suffix is either 'x', 'y', 'tx', 'ty', or 'graph'.
        if suffix not in names:
            continue
        datasets[suffix] = cPickle.load(open(fullpath, 'rb'), encoding="latin1")

    return datasets


class Dataset(data.Dataset):

    def __init__(self, directory, dataset_name, dataset_style, train=False):
        dataset = load_planetoid_data(directory, dataset_name, dataset_style)
        if train:
            self.x = dataset['x']
            self.y = dataset['y']
        else:
            self.x = dataset['tx']
            self.y = dataset['ty']
        self.graph = dataset['graph']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        self.x[idx], self.y[idx], self.graph[idx]
