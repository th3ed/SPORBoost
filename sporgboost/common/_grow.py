from numba import prange, deferred_type, optional, float64
from numba.experimental import jitclass
from sporgboost.common import best_split, gini_impurity
from .._arrays import row_mean, row_nunique
import numpy as np
from numba import njit

# node_type = deferred_type()

# # Node needs to be explicitly included in each tree type for numba
# # to properly compile
# @jitclass([
#     ('value', optional(float64[:,:])),
#     ('left', optional(node_type)),
#     ('right', optional(node_type)),
#     ('proj', optional(float64[:,:])),
#     ('split', optional(float64))
# ])
# class Node():
#     def __init__(self):
#         self.value = None
#         self.left = None
#         self.right = None
#         self.proj = None
#         self.split = None

#     def is_leaf(self):
#         return self.left is None and self.right is None

#     def init_children(self):
#         self.left = Node()
#         self.right = Node()
        
# node_type.define(Node.class_type.instance_type)

@njit(parallel=True, cache=False)
def _grow_tree(X, y, proj, max_depth, *args):
    # Initialize root of tree
    root = {}

    # Each piece of work contains a pointer to 
    # the node being processed and the index positions
    # of obs at that node
    nodes = [root]
    node_train_idx = [np.arange(0, X.shape[0])]

    depth = 0
    start = 0
    end = 1
    max_depth = np.inf if max_depth is None else max_depth

    while (depth <= max_depth) and ((end - start) > 0):
        # Parallel loop over all nodes to be processed
        nodes_added_in_round = 0
        for node_idx in range(start, end):
            # Get node and asociated obs
            node, idx = nodes[node_idx], node_train_idx[node_idx]
            X_, y_ = X[idx, :], y[idx, :]

            # Step 1: Check if node is a leaf
            node['value'] = row_mean(y_).reshape((1, -1))

            # Leaf check 1: at max depth
            if depth == max_depth:
                continue

            # Leaf check 2: partition is pure
            if gini_impurity(node['value']) == 0.:
                continue

            # Step 2: If node is not a leaf, find a split
            # Project data based on function
            A = proj(X_, *args)

            X_proj = X_ @ A

            # Leaf check 3: partition has no unique levels in X, can't
            # be partitioned further to improve performance
            if np.all(row_nunique(X_proj) <= 1):
                continue

            # Evaluate each col and candidate split
            col, node['split'] = best_split(X_proj, y_)
            node['proj'] = np.ascontiguousarray(A[:, col]).reshape((-1, 1))

            # Initalize children and add to the next iteration to be processed
            node['left'] = len(nodes)
            node['right'] = node['left'] + 1

            # Get idx arrays for the split
            # le = (X_proj[:, col] <= node['split'])
            nodes.append({})
            node_train_idx.append(idx[(X_proj[:, col] <= node['split'])])
            nodes.append({})
            node_train_idx.append(idx[~(X_proj[:, col] <= node['split'])])
            nodes_added_in_round += 2
        
        # Once new nodes have been added, increment the index and continue adding nodes if needed
        start = end
        end = end + nodes_added_in_round
        depth += 1
    return nodes
