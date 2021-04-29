from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np


NUM_ESTIMATORS = [50, 100, 200]
LEARNING_RATES = [0.1, 0.2]
MAX_DEPTH = [2, 5, 10]


def to_cls(y):
    return np.argmax(y, axis=1)

def run_brf_experiments(train_dataset, test_dataset):
    for n_estimators in NUM_ESTIMATORS:
        for lr in LEARNING_RATES:
            for depth in MAX_DEPTH:
                #print(f"num_estimators = {n_estimators}, learning_rate = {lr}, max_depth = {depth}")
                run_single_experiment(n_estimators, lr, depth, train_dataset, test_dataset)

def run_single_experiment(n_estimators, lr, depth, train_dataset, test_dataset):
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=depth, random_state=0)
    X, y = train_dataset.x, train_dataset.y
    #X_val, y_val = X[:20], y[:20]
    #X, y = X[20:], y[20:]
    clf.fit(X, to_cls(y))
    #print(clf.score(X_val, to_cls(y_val)))
    
    X, y = test_dataset.x, test_dataset.y
    print(clf.score(X, to_cls(y)))
