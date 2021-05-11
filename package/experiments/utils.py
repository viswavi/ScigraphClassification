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

def generate_reindexed_graph(graph, train_size, test_size, test_indices):
    index_remapping = {}
    for train_idx in range(train_size):
        index_remapping[train_idx] = train_idx
    for test_idx in range(test_size):
        index_remapping[test_indices[test_idx]] = test_idx + train_size

    reindexed_graph = {}
    for node_idx, neighbor_idxs in graph.items():
        if node_idx not in index_remapping:
            # Don't add any nodes that are completely unlabeled (i.e. not in train or test sets)
            # for now.
            continue
        remapped_node_idx  = index_remapping[node_idx]
        reindexed_graph[remapped_node_idx] = []

        for neighbor_idx in neighbor_idxs:
            if neighbor_idx not in index_remapping:
                # Ignore unlabeled data points.
                # TODO: support training on unlabeled points.
                continue
            reindexed_graph[remapped_node_idx].append(index_remapping[neighbor_idx])
    return reindexed_graph

# Convert all the different data sources (X_train, X_test) into a single matrix, and update
# the citation graph to have the node indices match this single matrix.
# Expects the first four arguments to be Torch tensors.
def consolidate_data(X_train, y_train, X_test, y_test, test_indices, graph):
    aggregated_X = torch.cat([X_train, X_test])
    aggregated_y = torch.cat([y_train, y_test])
    num_train = len(X_train)
    num_test = len(X_test)
    reindexed_graph = generate_reindexed_graph(graph, num_train, num_test, test_indices)
    return aggregated_X, aggregated_y, reindexed_graph
