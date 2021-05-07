import torch
import torch.nn as nn
from .mlp import MLP, make_multi_layer_perceptron



def unary_potential_product(unary_potentials):
    # Product is in log-space.
    unary_sum = None
    for p in unary_potentials:
        if unary_sum is None:
            unary_sum = p.clone()
        else:
            unary_sum = unary_sum + p
    return unary_sum

    
class TreeNLLLoss(nn.Module):
    def __init__(self):
        super(TreeNLLLoss,self).__init__()

    # def forward(self, log_likelihood):
    #     # ...implement the forward function for negative log likelihood loss...
    #     return -log_likelihood
    def forward(self, traversal_list, partition_func):
        # ...implement the forward function for negative log likelihood loss...
        losses = []
        loss = 0
        for node in traversal_list:
            true_parent = node.true_label
            for child in node.children:
                true_child = child.true_label
                edge_pot = child.parent_edge_potential[0,len(tag_dict)*true_child+true_parent]
                losses.append(torch.clone(edge_pot))
                losses.append(torch.clone(child.unary_potential[0, true_child]))
                loss = loss + edge_pot
                loss = loss + child.unary_potential[0, true_child]
                
            # b = norm_beliefs[i]
            # lbl = labels[i]
            # loss = -1*b[lbl] + torch.logsumexp(b)
            if node.parent is None: 
                losses.append(torch.clone(node.unary_potential[0, node.true_label]))
                loss = loss + node.unary_potential[0, node.true_label]
        if len(losses) == 1:
            losses = losses[0]
        else:
            losses = torch.cat(losses, dim=0)
        return -1*(loss-partition_func)


class TreeCRF(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers = 2):
        super(TreeCRF,self).__init__()

        self.build_unary_potentials = nn.Linear(hidden_dim, num_classes)
        self.build_edge_potentials = nn.Linear(2 * hidden_dim, num_classes * num_classes)

        self.criterion = TreeNLLLoss()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.mlp = make_multi_layer_perceptron(in_dim=input_dim, out_dim=None, hidden_dim=hidden_dim, num_layers=num_layers, feature_extractor=True)

    def build_traversal_and_get_features(self, node, traversal_list = [], features = []):
        if len(node.children) > 0 and node not in traversal_list:
            for child in node.children:
                if len(child.children) > 0 and child not in traversal_list:
                    traversal_list, features = self.build_traversal_and_get_features(child, traversal_list, features)
            if node not in traversal_list:
                traversal_list.append(node)
                features.append(node.features)
            return traversal_list, features

    def predict(self, norm_beliefs, labels, node_idxs):
        full = torch.cat(norm_beliefs, dim=0)

        preds = torch.argmax(full, axis = 1)
        mask = np.array([(p==a) for p,a in zip(preds, labels)]).astype(int)
        tree_acc = mask.sum()/len(mask)
        leaf_mask = mask[leaf_idxs]
        leaf_acc = leaf_mask.sum()/len(leaf_idxs)
        preds_dict = dict(zip(node_idxs, preds))
        return tree_acc, leaf_acc, preds_dict

    def forward(self, tree):
        if len(tree.children) == 0:
            single_features = torch.unsqueeze(tree.features, dim=0)
            tree.hidden = torch.clone(self.mlp(single_features))
            # Case where node has no neighbors.
            unary_pot = torch.clone(self.build_unary_potentials(tree.hidden))
            tree.unary_potential = unary_pot
            return [tree]

        traversal_list, features = self.build_traversal_and_get_features(tree, [], [])
        features_tensor = torch.stack(features)
        try:
            hiddens = self.mlp(features_tensor)
        except:
            breakpoint()
        
        for node, hidden in zip(traversal_list, hiddens):
            children = node.children
            
            node.hidden = torch.clone(hidden)

            unary_pot = torch.clone(self.build_unary_potentials(node.hidden))
            node.unary_potential = unary_pot
            
            for child in children:
                assert child.hidden is not None, breakpoint()
                # parent_features = node.features[:self.hidden]
                # parent_neighbors = node.features[self.hidden:]
                # child_features = child.features[:self.hidden]
                # child_neighbors = child.features[self.hidden:]
                # shared_neighbor_features = parent_neighbors + child_neighbors - parent_features - child_features

                #edge_pot = torch.clone(self.build_edge_potentials(torch.cat([parent_features, child_features, shared_neighbor_features], dim=1)))
                edge_pot = torch.clone(self.build_edge_potentials(torch.cat([node.hidden, child.hidden], dim=1)))
                child.parent_edge_potential = edge_pot
        return traversal_list 

####################################################################
## Belief propagation
####################################################################

    def belief_propagation(self, traversal_list):
        # ...implement belief propagation and return the beliefs for every node...
        norm_beliefs = []
        node_idxs = []

        labels = []
        sanity_check = None
        loss = 0
        for node in traversal_list:
            # do forward pass
            for child in node.children:
                child_pot = torch.clone(child.unary_potential)
                inc_msgs = []
                if len(child.children) > 0: 
                    for cc in child.children:
                        inc_msgs.append(torch.clone(cc.message_to_parent))
                inc_msgs.append(child_pot)
                tmp_ = torch.sum(torch.cat(inc_msgs, dim = 0), dim=0)
                tmp_ = tmp_.view(1, tmp_.shape[0])
                child.message_at = torch.clone(tmp_)
                # in fwd, take transpose 
                edge_pot = torch.clone(torch.transpose(child.parent_edge_potential.view(self.num_classes,self.num_classes), 0, 1))
                tmp_ =torch.cat(edge_pot.shape[0]*[tmp_])
                msg_to_parent = torch.logsumexp(edge_pot+tmp_, 1)
                child.message_to_parent = torch.clone(msg_to_parent).view(1, msg_to_parent.shape[0])
        
        for node in traversal_list[-1::-1]:
            # do backward pass (and compute beliefs??)
            belief_tmp = [torch.clone(node.unary_potential)]
            if node.parent is not None:
                assert node.message_from_parent is not None
                belief_tmp.append(torch.clone(node.message_from_parent))
            
            for i in range(len(node.children)): 
                child = node.children[i]
                assert child.message_to_parent is not None
                belief_tmp.append(torch.clone(child.message_to_parent))
                
                msg_from_p = [torch.clone(node.unary_potential)]
                if node.parent is not None:
                    assert node.message_from_parent is not None
                    msg_from_p.append(torch.clone(node.message_from_parent))
                if len(node.children) > 1:
                    
                    child2 = node.children[i+1-2*i]
                    assert child2.message_to_parent is not None
                    msg_from_p.append(torch.clone(child2.message_to_parent))
                # messages at parent 
                tmp_ = torch.sum(torch.cat(msg_from_p, dim = 0), dim=0)
                tmp_ = tmp_.view(1, tmp_.shape[0])
                assert child.message_at is not None
                msg_at_child = torch.repeat_interleave(torch.clone(child.message_at), repeats=tmp_.shape[1], dim=1)
                edge_tmp = child.parent_edge_potential

                #edge_belief = torch.sum(torch.cat([torch.cat(tmp_.shape[1]*[tmp_],dim=1), msg_at_child, edge_tmp], dim=0), dim=0)
                
                #print(torch.logsumexp(edge_belief, dim=0))
                edge_pot = torch.clone(edge_tmp.view(len(tag_dict),len(tag_dict)))
                tmp_ =torch.cat(edge_pot.shape[0]*[tmp_])
                msg_from_parent = torch.logsumexp(edge_pot+tmp_, 1)
                child.message_from_parent = torch.clone(msg_from_parent).view(1, msg_from_parent.shape[0])
                if len(child.children) == 0:
                    child_belief_tmp = torch.sum(torch.cat([child.unary_potential, child.message_from_parent], dim=0), dim=0)
                    assert torch.isclose(torch.logsumexp(child_belief_tmp, 0), sanity_check)
                    child.belief = torch.clone(child_belief_tmp)
                    norm_beliefs.append((child_belief_tmp-sanity_check).view(1, child_belief_tmp.shape[0]))
                    labels.append(child.true_label)
                    
            belief_tmp = torch.sum(torch.cat(belief_tmp, dim = 0), dim=0)
            partition_func = torch.logsumexp(belief_tmp, dim=0)
            #assert torch.isclose(torch.logsumexp(edge_belief, 0), partition_func)
            if sanity_check is not None:
                assert torch.isclose(partition_func, sanity_check)
            else:
                sanity_check = torch.clone(partition_func)
            node.belief = torch.clone(belief_tmp)

            norm_beliefs.append((belief_tmp-partition_func).view(1, belief_tmp.shape[0]))
            labels.append(node.true_label)
            node_idxs.append(node.idx)
        
        return norm_beliefs, labels, node_idxs, sanity_check
