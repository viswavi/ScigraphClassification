import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report


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


def run_graph_mlp_experiments(train_dataset, test_dataset):
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

    for hidden_dim in HIDDEN_DIMS:
        for lr in LEARNING_RATES:
            for num_layers in NUM_LAYERS:
                #print(f"hidden_dim = {hidden_dim}, learning_rate = {lr}, num_layers = {num_layers}")
                #run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test, X_val, y_val)
                run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test)


def run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    model = train_model(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test)


def train_model(hidden_dim, lr, num_layers, num_cls, X, y, X_val=None, y_val=None):
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


