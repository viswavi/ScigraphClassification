import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super(MLP, self).__init__()
        self.model = make_multi_layer_perceptron(in_dim, out_dim, hidden_dim, num_layers)

    def forward(self, features):
        return self.model(features)

def make_multi_layer_perceptron(in_dim, out_dim, hidden_dim, num_layers, feature_extractor=False):
    fc_1 = nn.Linear(in_dim, hidden_dim, bias=True)
    layers = [fc_1, nn.ReLU()]

    for _ in range(num_layers - 2):
        fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        layers.append(fc)
        layers.append(nn.ReLU())
    if not feature_extractor:
        out_layer = nn.Linear(hidden_dim, out_dim)
        layers.append(out_layer)
        layers.append(nn.Softmax())

    return nn.Sequential(*layers)