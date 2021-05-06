from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from package.datasets.graph_loader import load_graph_from_dataset

from package.experiments.run_graph_mlp_experiments import aggregate_features

DEVICE = 'cpu'
NUM_LAYERS = [2]
HIDDEN_DIMS = [100]
LEARNING_RATES = [0.03]
NUM_EPOCHS = 10


def to_cls(y):
    return np.argmax(y, axis=1)


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super(MLP, self).__init__()
        self.model = make_multi_layer_perceptron(in_dim, out_dim, hidden_dim, num_layers)

    def forward(self, features):
        return self.model(features)

def make_multi_layer_perceptron(in_dim, out_dim, hidden_dim, num_layers):
    fc_1 = nn.Linear(in_dim, hidden_dim, bias=True)
    layers = [fc_1, nn.ReLU()]

    for _ in range(num_layers - 2):
        fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        layers.append(fc)
        layers.append(nn.ReLU())

    out_layer = nn.Linear(hidden_dim, out_dim)
    layers.append(out_layer)
    layers.append(nn.Softmax())

    return nn.Sequential(*layers)

def generate_reindexed_graph(graph, all_X, train_size, test_size, test_indices):
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

# CRFs are undirected graphical models, so the underlying citation graph must be made undirected.
def make_graph_undirected(graph):
    undirected_graph = defaultdict(set)
    for s in graph:
        if graph[s] == []:
            undirected_graph[s] = set()
        for t in graph[s]:
            undirected_graph[s].add(t)
            undirected_graph[t].add(s)
    for k in undirected_graph:
        undirected_graph[k] = list(undirected_graph[k])
    return undirected_graph

def run_tree_mlp_experiments(train_dataset, test_dataset):
    X_train, y_train, all_X, graph = train_dataset.x, train_dataset.y, train_dataset.all_x, train_dataset.graph
    
    X_train = torch.tensor(X_train.toarray(), device=DEVICE)
    all_X = torch.tensor(all_X.toarray(), device=DEVICE)
    X_train = aggregate_features(X_train, all_X, graph)
    _, num_cls = y_train.shape
    y_train = torch.tensor(to_cls(y_train), device=DEVICE, dtype=torch.int64)
    
    #X_val, y_val = X_train[:20], y_train[:20]
    #X_train, y_train = X_train[20:], y_train[20:]

    X_test, y_test, test_indices = test_dataset.x, test_dataset.y, test_dataset.test_indices
    X_test = torch.tensor(X_test.toarray(), device=DEVICE)
    X_test = aggregate_features(X_test, all_X, graph, indices=test_indices)
    y_test = torch.tensor(to_cls(y_test), device='cpu', dtype=torch.int64)

    aggregated_X = torch.cat([X_train, X_test])
    aggregated_y = torch.cat([y_train, y_test])
    num_train = len(X_train)
    num_test = len(X_test)

    reindexed_graph = generate_reindexed_graph(graph, all_X, num_train, num_test, test_indices)
    undirected_graph = make_graph_undirected(reindexed_graph)
    all_edges = [v for vv in undirected_graph.values() for v in vv]


    train_loader, test_loader = load_graph_from_dataset(aggregated_X, aggregated_y, num_train, num_test, undirected_graph)


    for hidden_dim in HIDDEN_DIMS:
        for lr in LEARNING_RATES:
            for num_layers in NUM_LAYERS:
                run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test)


def run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    model = train_model(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test)


def train_model(hidden_dim, lr, num_layers, num_cls, X, y, X_val=None, y_val=None, use_crf = True):
    _, num_features = X.shape
    if not use_crf:
        model = MLP(in_dim=num_features, hidden_dim=hidden_dim, out_dim=num_cls, num_layers=num_layers)
    else:
        raise NotImplementedError
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(NUM_EPOCHS):

        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    if X_val is not None:
        y_pred = torch.argmax(model(X_val), dim=1)
        val_score = accuracy_score(y_val.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        #print(f"Validation accuracy {val_score}")
        print(val_score)
    return model


def evaluate_model(model, X, y):
    y_pred = torch.argmax(model(X), dim=1)
    y_pred = y_pred.cpu().numpy()

    print(f"Test accuracy {accuracy_score(y, y_pred)}")
