from package.datasets.dataset import Dataset

class Node():
    def __init__(self, idx, features, true_label, parent=None, children=[]):
        self.features = features

        self.true_label = true_label
        self.parent = parent
        self.children = children
        self.idx=idx
        self.unary_potential = None
        self.parent_edge_potential = None
        self.hidden= None


def construct_graph_neighborhoods(aggregated_X, aggregated_y, candidate_nodes, undirected_graph, max_neighborhood_size=10, inclusive=True):
    # if inclusive = True, then when constructing each candidate node's graph neighborhood, we only will include
    # nodes in the candidate set (e.g. only include other test nodes in the neighborhood of any test node).
    graph_neighborhoods = []
    for c in candidate_nodes:
        root_node = Node(c,
                        aggregated_X[c],
                        aggregated_y[c],
                        parent=None,
                        children=[])
        nodes = {c: root_node}
        search_queue = [(c, trg) for trg in undirected_graph[c]]
        # Keep a list of seen nodes to keep the graph acylic.
        seen_nodes = set([c])
        
        while len(search_queue) > 0 and len(nodes) <= max_neighborhood_size:
            [parent, search_node] = search_queue.pop(0)
            # If the new search node
            if search_node in seen_nodes:
                continue
            if search_node not in candidate_nodes and inclusive:
                continue

            seen_nodes.add(search_node)

            new_node = Node(search_node,
                            aggregated_X[search_node],
                            aggregated_y[search_node],
                            parent=parent,
                            children=[])
            
            nodes[parent].children.append(new_node)
            nodes[search_node] = new_node
            
            new_queue_nodes = [(search_node, trg) for trg in undirected_graph[search_node]]
            search_queue.extend(new_queue_nodes)

        graph_neighborhoods.append(root_node)
    return graph_neighborhoods


class GraphLoader():
    def __init__(self, aggregated_X, aggregated_y, candidate_nodes, undirected_graph, max_neighborhood_size=10, inclusive=True):
        self.trees = construct_graph_neighborhoods(aggregated_X, aggregated_y, candidate_nodes, undirected_graph,\
            max_neighborhood_size=max_neighborhood_size, inclusive=inclusive)

        self.ptr = 0

    def reset(self):
        self.ptr = 0


    def get_next_batch(self):
        current_batch = self.trees[self.ptr]

        # When you've returned the last tree, return done = True
        done = (self.ptr == len(self.trees) - 1)

        self.ptr += 1
        return done, current_batch

def aggregate_nodes_from_tree(tree):
    accumulator = []
    accumulator.append(tree.idx)
    if len(tree.children) > 0:
        for child in tree.children:
            accumulator.extend(aggregate_nodes_from_tree(child))
    return accumulator

def load_graph_from_dataset(aggregated_X, aggregated_y, num_train, num_test, num_validation, undirected_graph, device='cpu', inclusive=True):
    train_node_ids = list(range(num_train))
    test_node_ids = [i + num_train for i in range(num_test)]
    train_loader = GraphLoader(aggregated_X, aggregated_y, train_node_ids, undirected_graph, inclusive=inclusive)

    # Construct the validation set from the training set, by holding out entire graph neighborhoods.
    validation_node_ids = []
    for t in train_loader.trees:
        for node_idx in aggregate_nodes_from_tree(t):
            if node_idx not in validation_node_ids:
                validation_node_ids.append(node_idx)
        if len(validation_node_ids) >= num_validation:
            break
    validation_node_ids = validation_node_ids[:num_validation]

    train_node_ids = list(set(train_node_ids) - set(validation_node_ids))
    print(f"train_node_ids: {train_node_ids}")

    train_loader = GraphLoader(aggregated_X, aggregated_y, train_node_ids, undirected_graph, inclusive=inclusive)
    val_loader = GraphLoader(aggregated_X, aggregated_y, validation_node_ids, undirected_graph, inclusive=inclusive)
    test_loader = GraphLoader(aggregated_X, aggregated_y, test_node_ids, undirected_graph, inclusive=inclusive)
    return train_loader, val_loader, test_loader