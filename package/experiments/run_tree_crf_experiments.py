from collections import defaultdict
import json
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from package.datasets.graph_loader import load_graph_from_dataset

from package.experiments.utils import to_cls, aggregate_features, consolidate_data
from .tree_crf import TreeCRF, TreeNLLLoss
from collections import Counter, defaultdict


DEVICE = 'cpu'
# ENSEMBLING = [False, True]
ENSEMBLING = [True]
NUM_LAYERS = [1, 2, 3]
HIDDEN_DIMS = [200, 300, 500]
# LEARNING_RATES = [0.03, 0.05, 0.1]
LEARNING_RATES = [0.03]
TRAIN_SIZES = [170, 400, 600] # This includes the set of validation samples.

NUM_EPOCHS = 10
VALIDATION_SAMPLES = 30


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

def run_tree_crf_experiments(train_dataset, test_dataset, ensemble):
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
    _, num_features = X_train.shape

    aggregated_X, aggregated_y, reindexed_graph = consolidate_data(X_train, y_train, X_test, y_test, test_indices, graph)
    undirected_graph = make_graph_undirected(reindexed_graph)

    all_edges = [v for vv in undirected_graph.values() for v in vv]
    total_data_size = len(X_train) + len(X_test)
    parameter_scores = {}

    print(f"Running grid search over ensembling, train sizes, hidden dimensions, and learning rates")
    for ensembling in ENSEMBLING:
        for train_size in TRAIN_SIZES:
            for hidden_dim in HIDDEN_DIMS:
                for lr in LEARNING_RATES:
                    for num_layers in NUM_LAYERS:
                        test_size = total_data_size - train_size
                        print(f"\ntrain_size: {train_size}, hidden_dim: {hidden_dim}, lr: {lr}, num_layers: {num_layers}, ensembling: {ensembling}")
                        train_loader, val_loader, _ = load_graph_from_dataset(aggregated_X, aggregated_y, train_size, test_size, VALIDATION_SAMPLES, undirected_graph)
                        validation_accuracy = run_single_experiment(hidden_dim, lr, num_layers, num_features, num_cls, train_loader, val_loader, ensembling)
                        parameter_key = {
                                            "ensembling": ensembling,
                                            "train_size": train_size,
                                            "hidden_dim": hidden_dim,
                                            "lr"        : lr,
                                            "num_layers": num_layers
                                        }
                        parameter_scores[str(parameter_key)] = validation_accuracy

    print(f"Parameter evaluations:\n{json.dumps(parameter_scores, indent=4)}")
    best_parameter = max(parameter_scores, key=parameter_scores.get)
    best_parameter = eval(best_parameter)
    print(f"Best parameter combination: {best_parameter}")
    # Convert string dictionary back to dictinoary

    final_train_size = best_parameter["train_size"]
    final_test_size = total_data_size - final_train_size

    # Generate test loader without holding out a validation set, this time.
    train_loader, _, test_loader = load_graph_from_dataset(aggregated_X, aggregated_y, final_train_size, final_test_size, 0, undirected_graph)
    test_accuracy = run_single_experiment(best_parameter["hidden_dim"],
                                          best_parameter["lr"],
                                          best_parameter["num_layers"],
                                          num_features,
                                          num_cls,
                                          train_loader,
                                          test_loader,
                                          best_parameter["ensembling"])
    print(f"Final Test Accuracy (on test set of size {final_test_size}): {test_accuracy}.")


def run_single_experiment(hidden_dim, lr, num_layers, num_features, num_cls, train_loader, test_loader, ensemble):
    start_time = time.perf_counter()
    model = train_model(train_loader, hidden_dim, lr, num_layers, num_features, num_cls)
    test_accuracy = evaluate_model(model, test_loader, ensemble)
    end_time = time.perf_counter()
    print(f"Single model took {round(end_time - start_time)} seconds to train.")
    return test_accuracy


def train_model(train_loader, hidden_dim, lr, num_layers, num_features, num_cls, loss_interval=40):
    model = TreeCRF(input_dim=num_features, hidden_dim=hidden_dim, num_classes=num_cls, num_layers=num_layers)
    
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = model.criterion
    losses = []
    tree_accs = []
    leaf_accs = []

    for _ in tqdm(range(NUM_EPOCHS)):
        train_loader.reset()
        # step thru batches...
        done = False
        while not done:
            done, tree = train_loader.get_next_batch()

            optimizer.zero_grad()
            traversal_list = model(tree)
            norm_beliefs, labels, node_idxs, partition_func = model.belief_propagation(traversal_list)
            loss = criterion(traversal_list, partition_func)
            ## what is labels ... ah, i had a node.true_label in the posTagger 
            ## probably need to change this
            # preds = model.predict(norm_beliefs, labels, leaf_idxs)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if len(losses) == loss_interval:
                print(f"Average loss from {loss_interval} points: {round(np.mean(losses), 5)}")
                losses = []

    # if X_val is not None:
    #     y_pred = torch.argmax(model(X_val), dim=1)
    #     val_score = accuracy_score(y_val.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    #     #print(f"Validation accuracy {val_score}")
    #     print(val_score)
    return model

def vote_pooling(labels):
    label_count = Counter(labels)
    [(most_common_label, _)] = label_count.most_common(1)
    return most_common_label


def evaluate_model(model, test_loader, ensemble=False):
    done = False
    node_labels = []
    node_predictions = []

    # For voting over all trees with a given node, if ensemble is True.
    all_predictions = defaultdict(list)
    test_node_indices = []

    neighborhood_sizes = []

    while not done:
        done, tree = test_loader.get_next_batch()
        node_labels.append(tree.true_label)
        root_node_idx = tree.idx
        traversal_list = model(tree)
        norm_beliefs, labels, node_idxs, partition_func = model.belief_propagation(traversal_list)
        preds_dict = model.predict(norm_beliefs, labels, node_idxs)
        neighborhood_sizes.append(len(preds_dict))
        if ensemble:
            for node_idx, node_pred in preds_dict.items():
                all_predictions[node_idx].append(node_pred)
        else:
            node_predictions.append(preds_dict[root_node_idx])

        test_node_indices.append(root_node_idx)
    if ensemble:
        for node_idx in test_node_indices:
            all_labels = all_predictions[node_idx]
            top_label = vote_pooling(all_labels)
            node_predictions.append(top_label)
    test_accuracy = accuracy_score(node_labels, node_predictions)
    print(f"Test accuracy {test_accuracy}")
    print(f"Average neighborhood size in training: {round(np.mean(neighborhood_sizes), 3)}")
    return test_accuracy
