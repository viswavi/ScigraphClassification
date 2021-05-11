
import torch
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from experiments.utils import consolidate_data, aggregate_features

DEVICE = 'cpu'


NUM_ESTIMATORS = [50, 100, 200]
LEARNING_RATES = [0.1, 0.2]
MAX_DEPTH = [2, 5, 10]


def to_cls(y):
    return np.argmax(y, axis=1)

def remap_dataset(train_dataset, test_dataset):
    X_train, y_train, all_X, graph = train_dataset.x, train_dataset.y, train_dataset.all_x, train_dataset.graph
    X_test, y_test, test_indices = test_dataset.x, test_dataset.y, test_dataset.test_indices
    
    X_train = torch.tensor(X_train.toarray(), device=DEVICE)
    y_train = torch.tensor(to_cls(y_train), device=DEVICE, dtype=torch.int64)
    X_test = torch.tensor(X_test.toarray(), device=DEVICE)
    y_test = torch.tensor(to_cls(y_test), device='cpu', dtype=torch.int64)
    
    aggregated_X, aggregated_y, reindexed_graph = consolidate_data(X_train, y_train, X_test, y_test, test_indices, graph)

    return aggregated_X[:600], aggregated_y[:600], aggregated_X[600:], aggregated_y[600:]

def run_brf_experiments(train_dataset, test_dataset, validate=False):
    X_train, y_train, X_test, y_test = remap_dataset(train_dataset, test_dataset)
    for n_estimators in NUM_ESTIMATORS:
        for lr in LEARNING_RATES:
            for depth in MAX_DEPTH:
                run_single_experiment(n_estimators, lr, depth, X_train, y_train, X_test, y_test, validate)

def run_single_experiment(n_estimators, lr, depth, X_train, y_train, X_test, y_test, validate=False):
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=depth, random_state=0)
    if validate:
        X_val, y_val = X_train[570:], y_train[570:]
        X_train, y_train = X_train[:570], y_train[:570]
    clf.fit(X_train, y_train)
    
    if validate:
        print(clf.score(X_val, y_val))
    else:
        print(clf.score(X_test, y_test))

