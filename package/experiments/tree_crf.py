import torch



def unary_potential_product(unary_potentials):
    # Product is in log-space.
    unary_sum = None
    for p in unary_potentials:
        if unary_sum is None:
            unary_sum = p.clone()
        else:
            unary_sum = unary_sum + p
    return unary_sum

def belief_propagation(root, leaves, node_map, unary_potentials, edge_potentials, agg="sum"):
    beliefs = {}

    # ...implement belief propagation and return the beliefs for every node...
    # initialize messages with leaves' potentials
    message_queue = []

    root_idx = root.idx
    starting_node_indices = [l.idx for l in leaves]
    starting_node_indices.extend([k for k in unary_potentials if k not in starting_node_indices and k != root.idx])
    for node_idx in starting_node_indices:
        unary_potential = unary_potentials[node_idx]
        # Initialize message queue with unary-factor-to-node messages for each factor.
        starting_msg = Message(mtype = F2V, factor_name=node_idx, variable_name=node_idx, potential=unary_potential)
        message_queue.append(starting_msg)

    going_up = True
    just_changed_directions = False
    historical_messages = []
    # Pass messages from leaves upwards. Do this with a bottom-up tree traversal,
    # by expanding hidden representations for parents whose children have already been expanded.
    while len(message_queue) > 0:
        in_message = message_queue.pop(0)
        if in_message.mtype == F2V:
            # Received message is a factor-to-variable message.
            in_msg_source_factor = in_message.factor_name
            # The target variable from the incoming F2V message becomes the source variable of the
            # outgoing V2F message.
            in_msg_target_variable = in_message.variable_name
            out_msg_source_variable = in_msg_target_variable

            out_msg_target_factors = []
            if going_up:
                if out_msg_source_variable == root_idx:
                    out_msg_target_factor = root_idx
                    out_msg_target_factors = [root_idx]
                else:
                    parent = node_map[out_msg_source_variable].parent
                    assert parent is not None
                    out_msg_target_factor = (parent.idx, out_msg_source_variable)
                    out_msg_target_factors = [out_msg_target_factor]
                pass
            else:
                unary_target_factor = out_msg_source_variable
                if unary_target_factor != root_idx:
                    # We have already used the variable-to-factor message for the root node.
                    # For all other nodes, let's add this to our queue of messages to populate.
                    out_msg_target_factors.append(unary_target_factor)
                for c in node_map[out_msg_source_variable].children:
                    # Send a message to each downstream factor.
                    out_msg_target_factor = (out_msg_source_variable, c.idx)
                    out_msg_target_factors.append(out_msg_target_factor)

            N_i = count_neighbors(node_map[out_msg_source_variable]) + 1
            postpone_message = False
            for out_msg_target_factor in out_msg_target_factors:
                # Looking for all F2V variables ending with `out_msg_source_variable`, except the one coming from the target factor.
                # N(i) := one factor for each neighbor of i, plus the unary factor at i
                matching_factors = [m for m in [in_message] + message_queue + historical_messages if \
                                        m.mtype == F2V  and m.variable_name == out_msg_source_variable  and m.factor_name != out_msg_target_factor]
                if len(matching_factors) != N_i-1:
                    # In order to propagate to the target factor, we must already have access to messages from all other factors.
                    # This should only ever happen during our upward propagation. Since this is a tree, during
                    # downward propagation, we should already have all necessary messages other than `message` from
                    # the upward pass. 
                    assert going_up, breakpoint()
                    postpone_message = True
                    break
                for m in matching_factors:
                    if m in message_queue:
                        message_queue.remove(m)
                        # Since we've removed the message from the message queue, place it on the historical messages list
                        # for future reference.
                        historical_messages.append(m)

                # Process factor, and create new message from `out_msg_source_variable` to `out_msg_target_factor`.
                if going_up and out_msg_target_factor == root_idx:                    # Final message in the upward chain. It may not be necessary, but lets place it in the historical
                    # messages store for consistency.
                    final_up_msg = Message(mtype = V2F,
                                                    factor_name=root_idx,
                                                    variable_name=root_idx,
                                                    potential=in_message.potential)
                    historical_messages.append(final_up_msg)
                    assert len(message_queue) == 0, breakpoint()
                    going_up = False
                    just_changed_directions = True
                    root_down_msg = Message(mtype = F2V,
                                                factor_name=root_idx,
                                                variable_name=root_idx,
                                                potential=unary_potentials[root_idx])
                    message_queue.append(root_down_msg)

                    # TODO(Vijay): replace the just_changed_directions with a simple `continue`
                else:
                    factor_potentials = [m.potential for m in matching_factors]
                    product_of_potentials = unary_potential_product(factor_potentials)
                    new_v2f = Message(mtype = V2F,
                        factor_name=out_msg_target_factor,
                        variable_name=out_msg_source_variable,
                        potential=product_of_potentials)
                    message_queue.append(new_v2f)

            # Compute node belief if making downward pass, since at this point we should have access to all necessary factor-to-variable messages.
            if going_up is False and not just_changed_directions:
                all_matching_factors = [m for m in [in_message] + message_queue + historical_messages if \
                    m.mtype == F2V  and m.variable_name == out_msg_source_variable]
                assert len(all_matching_factors) == N_i, breakpoint()

                factor_potentials = [m.potential for m in all_matching_factors]
                variable_belief = unary_potential_product(factor_potentials)
                beliefs[in_msg_target_variable] = variable_belief

            # Since we've removed the message from the message queue, place it on the historical messages list
            # for future reference.
            historical_messages.append(in_message)

            if just_changed_directions:
                just_changed_directions = False

            if postpone_message:
                # Place this message at the end of the queue, for when we're ready to use it.
                if in_message in historical_messages:
                    historical_messages.remove(in_message)
                message_queue.append(in_message)
        else:
            # Received message is a variable-to-factor message.
            in_msg_source_variable = in_message.variable_name
            # The target factor from the incoming V2F message becomes the source factor of the
            # outgoing F2V message.
            in_msg_target_factor = in_message.factor_name
            out_msg_source_factor = in_msg_target_factor

            historical_messages.append(in_message)
            if not isinstance(in_msg_target_factor, tuple):
                # Then this is a message from a node to a unary factor, which we only need to compute factor
                # beliefs. Therefore, just throw this on the historical messages record and forget about it.
                continue
            
            # Target factor consists of (parent, child)
            if going_up:
                # Message is being sent from child to factor with parent.
                assert in_msg_source_variable == in_msg_target_factor[1]
                out_msg_target_variable = in_msg_target_factor[0]
            else:
                # Message is being sent from parent to factor with child.
                assert in_msg_source_variable == in_msg_target_factor[0]
                out_msg_target_variable = in_msg_target_factor[1]
            edge_potential = edge_potentials[in_msg_target_factor]            

            factor_potential = edge_potential if going_up else edge_potential.T

            if agg == "sum":
                new_potential = torch.logsumexp(factor_potential + in_message.potential, dim=1)
            elif agg == "max":
                new_potential, _ = torch.max(factor_potential + in_message.potential, dim=1)
            else:
                raise ValueError("Algorithm must be sum-product or max-product.")

            new_f2v = Message(mtype = F2V,
                factor_name=out_msg_source_factor,
                variable_name=out_msg_target_variable,
                potential=new_potential)
            message_queue.append(new_f2v)

    assert len(message_queue) == 0
    # 1 message from each unary factor to each node and vice-versa (total of 2 per node).
    # 2 messages from each node to each neighbor via their pairwise factor, and vice-versa (total of 4 per edge).
    # n nodes and n-1 edges -> 2*n + 4*(n-1) messages.
    assert len(historical_messages) == 2*len(node_map) + 4*(len(node_map)-1), breakpoint()
    assert len(beliefs) == len(node_map)
    if agg == "sum":
        # Check that all partition function values are the same, from any belief.
        belief_sums = [torch.logsumexp(v, dim=0).item() for v in beliefs.values()]
        for b in belief_sums:
            assert np.isclose(b, belief_sums[0], rtol=1e-3)

    return beliefs

class TreeNLLLoss(nn.Module):
    def __init__(self):
        super(TreeNLLLoss,self).__init__()

    def forward(self, log_likelihood):
        # ...implement the forward function for negative log likelihood loss...
        return -log_likelihood


class TreeCRF(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super(POSTagger,self).__init__()

        self.build_unary_potentials = nn.Linear(2*hidden_dim, num_classes)
        self.build_edge_potentials = nn.Linear(3 * hidden_dim, num_classes * num_classes)

        self.criterion = TreeNLLLoss()
        self.num_classes = num_classes

    def forward(self, sentence, tree, tree_len):
        # ...implement the forward function...

        embeds = self.embedding(sentence)
        hidden, _ = self.lstm(torch.unsqueeze(embeds, 1))
        hidden = torch.squeeze(hidden, 1)

        leaves, node_map = get_nodes(tree)
        assert len(leaves) == len(hidden)

        unary_potentials = {}
        edge_potentials = {}
        true_label_potential = None

        search_queue = []
        search_index_set = set()
        for i, leaf in enumerate(leaves):
            tag_probabilities = self.build_unary_potentials(hidden[i])
            unary_potentials[leaf.idx] = (hidden[i], tag_probabilities)
            if true_label_potential is None:
                true_label_potential = tag_probabilities[leaf.true_label[0]].clone()
            else:
                true_label_potential = true_label_potential + tag_probabilities[leaf.true_label[0]]
            if leaf.parent.idx not in search_index_set:
                search_queue.append(leaf.parent)
                search_index_set.add(leaf.parent.idx)
            assert leaf.children == []

        # In reverse, expand hidden representations for parents whose children have already
        # been expanded.
        while len(search_queue) > 0:
            query_node = search_queue.pop(0)
            search_index_set.remove(query_node.idx)
            expandable = False
            if len(query_node.children) == 1:
                assert query_node.children[0].idx in unary_potentials
                expandable = True
            else:
                assert len(query_node.children) == 2
                expandable = True
                for c in query_node.children:
                    if c.idx not in unary_potentials:
                        expandable = False
                        break
            if not expandable:
                # Wait until we've expanded the other children of this node
                search_queue.append(query_node)
                search_index_set.add(query_node.idx)
                continue

            if len(query_node.children) == 1:
                child_hidden, _, = unary_potentials[query_node.children[0].idx]
                hidden_raw = torch.cat([child_hidden, child_hidden], dim=0)
            elif len(query_node.children) == 2:
                child_hidden_l, _, = unary_potentials[query_node.children[0].idx]
                child_hidden_r, _, = unary_potentials[query_node.children[1].idx]
                hidden_raw = torch.cat([child_hidden_l, child_hidden_r], dim=0)
            else:
                raise ValueError("Should be unreachable")

            hidden = self.build_intermediate_hidden(hidden_raw)
            unary_tag_probabilities = self.build_unary_potentials_intermediate_node(hidden_raw)
            unary_potentials[query_node.idx] = (hidden, unary_tag_probabilities)
            true_label_potential = true_label_potential + unary_tag_probabilities[query_node.true_label[0]]

            for c in query_node.children:
                child_hidden, _, = unary_potentials[c.idx]
                edge_hidden = torch.cat([hidden, child_hidden], dim=0)
                edge_tag_probabilities = self.build_edge_potentials(edge_hidden).view((self.num_tags, self.num_tags))
                edge_potentials[(query_node.idx, c.idx)] = edge_tag_probabilities
                true_label_potential = true_label_potential + edge_tag_probabilities[query_node.true_label[0]][c.true_label[0]]
            
            if query_node.parent is not None:
                # Add each node for which we have not already calculated its unary potentials, which
                # also means we haven't calculated any of its adjacent edge potentials.
                if query_node.parent.idx not in unary_potentials and query_node.parent.idx not in search_index_set:
                    search_queue.append(query_node.parent)
                    search_index_set.add(query_node.parent.idx)


        # No need to keep unary hidden states after passing up through tree.
        unary_potentials = {k:tag_probs for k, (hidden, tag_probs) in unary_potentials.items()}
        max_beliefs = belief_propagation(tree, leaves, node_map, unary_potentials, edge_potentials, agg="max")
        sum_beliefs = belief_propagation(tree, leaves, node_map, unary_potentials, edge_potentials, agg="sum")

        node_to_prediction = {}
        for m, max_belief in max_beliefs.items():
            node_to_prediction[m] = torch.argmax(max_belief)
        
        some_belief = list(sum_beliefs.values())[0]
        partition_function = torch.logsumexp(some_belief, dim=0)
        assert true_label_potential < partition_function

        # Assert that the best possible product of potentials across all variable assignments is lesser than
        # the partition function, as another sanity check.
        assert torch.max(max_belief) < partition_function
        return node_to_prediction, true_label_potential - partition_function