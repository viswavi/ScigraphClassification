import numpy as np 
import torch 

def to_cls(y):
    return np.argmax(y, axis=1)

def aggregate_features(X, all_X, graph, indices=None):
    new_X = []
    for idx in range(len(X)):
        feats = X[idx]
        if indices is None:
            graph_indices = [i for i in graph[idx] if i < len(all_X)]
        else:
            test_index = indices[idx]
            graph_indices = [i for i in graph[test_index] if i < len(all_X)]
        neighbor_feats = torch.sum(all_X[graph_indices], dim=0)
        new_X.append(torch.cat((feats, neighbor_feats)))
    return torch.stack(new_X)