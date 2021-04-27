'''
python load_planetoid_data.py --planetoid-directory /Users/vijay/Documents/classes/10-708/ScigraphClassification/data \
                              --dataset-name citeseer

'''
import argparse
from collections import defaultdict
import json
import numpy as np
import os
import pandas as pd
import pickle
import sys

sys.path.append(os.getcwd() + '/..')

from datasets.dataset import load_planetoid_data


def construct_in_test_graph(graph, tx, ty):
    test_indices = tx.indices
    set_test_indices = set(test_indices)
    graph_cited_set = set([v for valset in graph.values() for v in valset] + [int(k) for k in graph.keys()])
    num_test_indices = [t for t in test_indices if t in graph_cited_set]
    test_keys = [t for t in test_indices if t in graph]
    all_graph = {t: graph[t] for t in test_keys}
    test_graph = {t: [v for v in graph[t] if v in test_indices] for t in test_keys}

    avg_all_degree = sum([len(v) for v in all_graph.values()]) / float(len(all_graph))
    avg_in_test_degree = sum([len(v) for v in test_graph.values()]) / float(len(test_graph))

    return avg_all_degree, avg_in_test_degree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--planetoid-directory', type=str, required=False, default="/projects/ogma1/vijayv/planetoid/data")
    parser.add_argument('--dataset-name', type=str, choices=["citeseer", "pubmed", "cora"])
    parser.add_argument('--dataset-style', type=str, choices=["ind", "trans"], default="trans")
    args = parser.parse_args()
    planetoid_data = load_planetoid_data(args.planetoid_directory, args.dataset_name, args.dataset_style)
    graph_key = [k for k in planetoid_data if "graph" in k][0]
    test_x_key = [k for k in planetoid_data if "tx" in k][0]
    test_y_key = [k for k in planetoid_data if "ty" in k][0]
    avg_all_degree, avg_in_test_degree = construct_in_test_graph(planetoid_data[graph_key], planetoid_data[test_x_key], planetoid_data[test_y_key])
    print(f"Avg Test Degree: {avg_all_degree}")
    print(f"Avg Test In-Test Degree: {avg_in_test_degree}")

if __name__ == "__main__":
    main()
