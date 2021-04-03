from sklearn.ensemble import GradientBoostingClassifier
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
                run_single_experiment(n_estimators, lr, depth, train_dataset, test_dataset)

def run_single_experiment(n_estimators, lr, depth, train_dataset, test_dataset):
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=depth, random_state=0)
    X, y = train_dataset.x, train_dataset.y
    clf.fit(X, to_cls(y))
    
    X, y = test_dataset.x, test_dataset.y
    print(clf.score(X, to_cls(y)))
