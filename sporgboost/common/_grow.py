from sporgboost.common import best_split, gini_impurity
from .._arrays import row_mean, row_nunique
import numpy as np
from numba import njit, range

@njit(cache=True, fastmath=True)
def _grow_tree(X, y, proj, max_depth, *args):
    # Each piece of work contains a pointer to 
    # the node being processed and the index positions
    # of obs at that node
    node_train_idx = {1 : np.arange(0, X.shape[0])}
    
    # To track the information for the nodes and use numba
    # we need to keep homogenous data types, use dicts
    node_value = {}
    node_split = {}
    node_proj = {}

    depth = 0
    max_depth = np.inf if max_depth is None else max_depth

    while (depth <= max_depth) and len(node_train_idx) > 0:
        # Parallel loop over all nodes to be processed
        for _ in range(len(node_train_idx)):
            # Get node and asociated obs
            node_idx, idx = node_train_idx.popitem()

            X_, y_ = X[idx, :], y[idx, :]

            # Step 1: Check if node is a leaf
            node_value[node_idx] = row_mean(y_).reshape((1, -1))

            # Leaf check 1: at max depth
            if depth == max_depth:
                continue

            # Leaf check 2: partition is pure
            if gini_impurity(node_value[node_idx]) == 0.:
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
            col, node_split[node_idx] = best_split(X_proj, y_)
            node_proj[node_idx] = np.ascontiguousarray(A[:, col]).reshape((-1, 1))

            # Initalize children and add to the next iteration to be processed
            node_train_idx[node_idx * 10] = idx[(X_proj[:, col] <= node_split[node_idx])]
            node_train_idx[node_idx * 10 + 1] = idx[(X_proj[:, col] > node_split[node_idx])]
        depth += 1

    return node_value, node_split, node_proj
