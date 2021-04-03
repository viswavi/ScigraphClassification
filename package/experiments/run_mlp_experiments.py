import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import f1_score


HIDDEN_DIMS = [50, 100, 200]
NUM_LAYERS = [3, 6]
LEARNING_RATES = [1e-4, 1e-3]
NUM_EPOCHS = 5

DEVICE = 'cuda'

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
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def run_mlp_experiments(train_dataset, test_dataset):
    X_train, y_train = train_dataset.x, train_dataset.y
    X_train = torch.tensor(X_train.toarray(), device=DEVICE)
    _, num_cls = y_train.shape
    y_train = torch.tensor(to_cls(y_train), device=DEVICE, dtype=torch.int64)
    
    X_test, y_test = test_dataset.x, test_dataset.y
    X_test = torch.tensor(X_test.toarray(), device=DEVICE)
    y_test = torch.tensor(to_cls(y_test), device='cpu', dtype=torch.int64)
    for hidden_dim in HIDDEN_DIMS:
        for lr in LEARNING_RATES:
            for num_layers in NUM_LAYERS:
                run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test)

def run_single_experiment(hidden_dim, lr, num_layers, num_cls, X_train, y_train, X_test, y_test):
    model = train_model(hidden_dim, lr, num_layers, num_cls, X_train, y_train)
    evaluate_model(model, X_test, y_test)

def train_model(hidden_dim, lr, num_layers, num_cls, X, y):
    _, num_features = X.shape
    model = MLP(in_dim=num_features, hidden_dim=hidden_dim, out_dim=num_cls, num_layers=num_layers)
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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

    print(f1_score(y, y_pred, average=None))


