'''
python load_planetoid_data.py --planetoid-directory /Users/vijay/Documents/classes/10-708/ScigraphClassification/data \
                              --dataset-name cora
#                              --dataset-name {cora/citeseer/pubmed}

'''
import argparse
from collections import defaultdict
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys
from collections import Counter
sys.path.append(os.getcwd() + '/..')

from datasets.dataset import load_planetoid_data


def construct_in_test_graph(graph, x, tx, ty):
    train_indices = list(range(x.shape[0]))
    test_indices = [i + x.shape[0] for i in range(tx.shape[0])]
    set_test_indices = set(test_indices)
    graph_cited_set = set([v for valset in graph.values() for v in valset] + [int(k) for k in graph.keys()])
    num_test_indices = [t for t in test_indices if t in graph_cited_set]
    test_keys = [t for t in test_indices if t in graph]
    all_graph = {t: graph[t] for t in test_keys}
    test_graph = {t: [v for v in graph[t] if v in test_indices] for t in test_keys}
    test_to_train_graph = {t: [v for v in graph[t] if v in train_indices] for t in test_keys}

    avg_all_degree = sum([len(v) for v in all_graph.values()]) / float(len(all_graph))
    avg_in_test_degree = sum([len(v) for v in test_graph.values()]) / float(len(test_graph))
    avg_test_to_train_degree = sum([len(v) for v in test_to_train_graph.values()]) / float(len(test_graph))

    return avg_all_degree, avg_in_test_degree, avg_test_to_train_degree


def draw_bar_chart(x, y, xlabel=None, ylabel=None, title=None, fname="/tmp/scratch.png"):
    fig, ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    barWidth = 0.5
    # Set position of bar on X axis
    r1 = np.arange(len(x))
    # Make the plot
    plt.bar(r1, y, color='lightblue', width=barWidth, edgecolor='white', label='baseline')

    # Add xticks on the middle of the group bars
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([r for r in range(len(x))], x)

    # Create legend & Show graphic
    #plt.legend(loc='upper left')
    fig.savefig(fname, dpi=400, bbox_inches='tight')
    print(f"Saved plot to {fname}")
    plt.tight_layout()
    del fig


def plot_label_distributions(test_y, dataset_name):
    labels = list(range(test_y.shape[1]))

    label_counts = {i: sum(test_y[:, i]) for i in labels}
    fname = f"/tmp/{dataset_name}_label_distribution.png"
    draw_bar_chart(list(label_counts.keys()), list(label_counts.values()), xlabel = "Label", ylabel="Frequency", title=f"Label distribution of {dataset_name}", fname=fname)


def sample_graph_neighborhood(train_x, test_x, test_y, graph, neighborhood_size=8):
    train_indices = list(range(train_x.shape[0]))
    test_indices = [i + train_x.shape[0] for i in range(test_x.shape[0])]

    np.random.seed(0)
    while True:
        test_index = np.random.choice(test_indices)
        print(f"\nStarting with new seed node {test_index}")
        connections = {}
        neighborhood = {}
        next_up = [test_index]
        while len(neighborhood) <= neighborhood_size and len(next_up) > 0 :
            new_node = next_up.pop(0)
            if new_node in test_indices:
                neighborhood[new_node] = ("test", test_y[new_node - train_x.shape[0]])
            elif new_node in train_indices:
                neighborhood[new_node] = ("train", test_y[new_node - train_x.shape[0]])
            else:
                neighborhood[new_node] = ("unlabeled", None)
            if new_node in graph:
                if new_node not in connections:
                    connections[new_node] = []
                for next in graph[new_node]:
                    if next not in neighborhood:
                        print(f"Adding edge from {new_node} to {next}")
                        connections[new_node].append(next)
                        next_up.append(next)
        if len(neighborhood) >= neighborhood_size:
            label_set = [str(b) for a, b in neighborhood.values() if b is not None]
            if len(label_set) > 5 and len(set(label_set)) >= 2 and min(Counter(label_set).values()) >= 3:
                break

    pruned_connections = {k: [vv for vv in v if vv in neighborhood] for k, v in connections.items() if k in neighborhood}
    return neighborhood, pruned_connections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--planetoid-directory', type=str, required=False, default="/projects/ogma1/vijayv/planetoid/data")
    parser.add_argument('--dataset-name', type=str, choices=["citeseer", "pubmed", "cora"])
    parser.add_argument('--dataset-style', type=str, choices=["ind", "trans"], default="trans")
    args = parser.parse_args()
    planetoid_data = load_planetoid_data(args.planetoid_directory, args.dataset_name, args.dataset_style)
    graph = planetoid_data["graph"]
    train_x_data = planetoid_data["x"]
    test_x_data = planetoid_data["tx"]
    test_y_data = planetoid_data["ty"]
    avg_all_degree, avg_in_test_degree, avg_test_to_train_degree = construct_in_test_graph(graph, train_x_data, test_x_data, test_y_data)

    plot_label_distributions(planetoid_data["ty"], args.dataset_name)

    print(f"# of Nodes: {len(graph)}")
    print(f"# of Edges: {len([y for yy in graph.values() for y in yy])//2}")


    print(f"# of Train Samples: {train_x_data.shape[0]}")
    print(f"# of Test Samples: {test_x_data.shape[0]}")
    print(f"# of Features: {test_x_data.shape[1]}")
    print(f"# of Labels: {test_y_data.shape[1]}")

    print(f"Avg Test Degree: {avg_all_degree}")
    print(f"Avg Test In-Test Degree: {avg_in_test_degree}")
    print(f"Avg Test-to_Train Degree: {avg_test_to_train_degree}")

    sample_graph_neighborhood(train_x_data, test_x_data, test_y_data, graph)



if __name__ == "__main__":
    main()
