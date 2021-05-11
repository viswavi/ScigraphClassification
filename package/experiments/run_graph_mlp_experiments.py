import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .mlp import MLP, make_multi_layer_perceptron
from experiments.utils import consolidate_data, aggregate_features, to_cls
torch.manual_seed(0)


from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report


DEVICE = 'cpu'
NUM_LAYERS = [1, 2, 3]
HIDDEN_DIMS = [100, 200, 300, 500]
LEARNING_RATES = [0.03, 0.05, 0.1]
NUM_EPOCHS = 10


def remap_dataset(train_dataset, test_dataset):
    X_train, y_train, all_X, graph = train_dataset.x, train_dataset.y, train_dataset.all_x, train_dataset.graph
    X_test, y_test, test_indices = test_dataset.x, test_dataset.y, test_dataset.test_indices
    
    X_train = torch.tensor(X_train.toarray(), device=DEVICE)
    all_X = torch.tensor(all_X.toarray(), device=DEVICE)
    X_train = aggregate_features(X_train, all_X, graph)
    y_train = torch.tensor(to_cls(y_train), device=DEVICE, dtype=torch.int64)
    
    X_test = torch.tensor(X_test.toarray(), device=DEVICE)
    X_test = aggregate_features(X_test, all_X, graph, indices=test_indices)
    y_test = torch.tensor(to_cls(y_test), device='cpu', dtype=torch.int64)
    
    aggregated_X, aggregated_y, reindexed_graph = consolidate_data(X_train, y_train, X_test, y_test, test_indices, graph)

    return aggregated_X[:600], aggregated_y[:600], aggregated_X[600:], aggregated_y[600:]


def run_graph_mlp_experiments(train_dataset, test_dataset, validate):
    _, num_cls = train_dataset.y.shape
    X_train, y_train, X_test, y_test = remap_dataset(train_dataset, test_dataset)
    for hidden_dim in HIDDEN_DIMS:
        for lr in LEARNING_RATES:
            for num_layers in NUM_LAYERS:
                run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test, validate)


def run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test, validate):
    if validate:
        X_val, y_val = X_train[570:], y_train[570:]
        X_train, y_train = X_train[:570], y_train[:570]
    
    model = train_model(hidden_dim, lr, num_layers, num_cls, X_train, y_train)

    if validate:
        evaluate_model(model, X_val, y_val)
    else:
        evaluate_model(model, X_test, y_test)


def train_model(hidden_dim, lr, num_layers, num_cls, X, y):
    _, num_features = X.shape
    model = MLP(in_dim=num_features, hidden_dim=hidden_dim, out_dim=num_cls, num_layers=num_layers)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(NUM_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, X, y):
    y_pred = torch.argmax(model(X), dim=1)
    y_pred = y_pred.cpu().numpy()

    print(accuracy_score(y, y_pred))


