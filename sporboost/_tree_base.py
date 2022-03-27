from numba import njit
import numpy as np
from ._arrays import row_cumsum, collapse_levels, row_mean, row_nunique, row_norm

@njit(cache=True, fastmath=True)
def gini_impurity(y):
    ''' Computes gini impunity for a given partition of data
    
    Args:
        y (Array): 2-d vector of normalized 
    '''
    return 1 - (y ** 2).sum(axis=1)

@njit(cache=True, fastmath=True)
def _grow_tree(X, y, proj, max_depth, sample_weight, *args):
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

            X_, y_, n_ = X[idx, :], y[idx, :], sample_weight[idx].reshape((-1, 1))

            # Step 1: Check if node is a leaf
            # The row mean calc doesn't always generate true probs due to a 
            # loss of precision multiplying and dividing by n, fix that here
            node_value[node_idx] = row_norm(row_mean(y_, n_).reshape((1, -1)))

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
            col, node_split[node_idx] = best_split(X_proj, y_, n_)
            node_proj[node_idx] = np.ascontiguousarray(A[:, col]).reshape((-1, 1))

            # Initalize children and add to the next iteration to be processed
            node_train_idx[node_idx * 10] = idx[(X_proj[:, col] <= node_split[node_idx])]
            node_train_idx[node_idx * 10 + 1] = idx[(X_proj[:, col] > node_split[node_idx])]
        depth += 1

    return node_value, node_split, node_proj

@njit(cache=True, fastmath=True)
def _predict_tree(tree_value, tree_split, tree_proj, X, n_classes):
    pred_mat = np.eye(n_classes)
    out = np.zeros(shape=(X.shape[0], n_classes))

    node_eval = {1 : np.arange(X.shape[0])}

    # Iteratively evaluate rules until each partition reaches a leaf
    while len(node_eval) > 0:
        idx_node, idx_rows = node_eval.popitem()
        X_ = X[idx_rows, :]

        # Are we at a leaf?
        if idx_node not in tree_split:
            # Reached a leaf, set value

            out[idx_rows, :] = pred_mat[np.argmax(tree_value[idx_node]), :]
            continue
        
        # Decision Stump, eval split and keep iterating
        X_proj = X_ @ tree_proj[idx_node]
        le = (X_proj <=tree_split[idx_node]).flatten()

        # Add child nodes to the stack to continue evaluation
        node_eval[idx_node * 10] = idx_rows[le]
        node_eval[idx_node * 10 + 1] = idx_rows[~le]
    return out

@njit(cache=True, fastmath=True)
def _predict_proba_tree(tree_value, tree_split, tree_proj, X, n_classes):
    out = np.zeros(shape=(X.shape[0], n_classes))

    node_eval = {1 : np.arange(X.shape[0])}

    # Iteratively evaluate rules until each partition reaches a leaf
    while len(node_eval) > 0:
        idx_node, idx_rows = node_eval.popitem()
        X_ = X[idx_rows, :]

        # Are we at a leaf?
        if idx_node not in tree_split:
            # Reached a leaf, set value

            out[idx_rows, :] = tree_value[idx_node]
            continue
        
        # Decision Stump, eval split and keep iterating
        X_proj = X_ @ tree_proj[idx_node]
        le = (X_proj <=tree_split[idx_node]).flatten()

        # Add child nodes to the stack to continue evaluation
        node_eval[idx_node * 10] = idx_rows[le]
        node_eval[idx_node * 10 + 1] = idx_rows[~le]
    return out

@njit(cache=True, fastmath=True)
def best_split(X, y, n):
    col_split_gini = find_split(X, y, n)
    
    # Once all cols have been tested, determine which made the best split
    col_idx = np.argmin(col_split_gini[:,1])
    return (col_idx, col_split_gini[col_idx, 0])

@njit(cache=True, fastmath=True)
def find_split(X, y, n):
    # Evaluate best split among each feature
    # 2d array where rows correspond to each col, 1st col is
    # split value and 2nd is gini
    col_split_gini = np.empty(shape = (X.shape[1], 2))
    for i in range(0, X.shape[1]):
        col_split_gini[i, :] = _find_split_feat(X[:, i], y, n)
    return col_split_gini

@njit(cache=True, fastmath=True)
def _find_split_feat(X, y, n):
    '''Determines where a split should be placed along a 1-d continuous feature col wrt y
    
    Args:
        X (Array): 1-d feature column to evalute for a split
        y (Array): 1-d response column
        
    Returns:
        A tuple of the split value for X and it's associated gini impunity
    '''
    # Step 1: Get unique levels of X, counts and sums of y ordered by X
    # This solves problems later where we need the pair of X, y sorted
    # along X but also want to force the algorithm splits to consider
    # a split has to include all rows that match the level, not just
    # the one that mins gini
    X_, y_, n_ = collapse_levels(X, y, n)

    # Bail if X_ has only one level
    if X_.shape[0] == 1:
        return np.NaN, 999
    
    # Step 2: Compute the prediction for y if we made the split at the given row
    # Note we will remove the last level from these arrays as we can't split
    # inclusively on this and have obs in the right partition
    n_total = n_.sum()
    y_total = y_.sum(axis=0)
    y_asc_cumsum = row_cumsum(y_)[:-1]
    y_desc_cumsum = y_total - y_asc_cumsum
    n_asc_cumsum = n_.cumsum()[:-1]
    n_desc_cumsum = n_total - n_asc_cumsum

    y_pred_left = (y_asc_cumsum / n_asc_cumsum.reshape((-1, 1)))
    y_pred_right = (y_desc_cumsum / n_desc_cumsum.reshape((-1, 1)))

    # Step 3: Compute gini impunity for left and right
    gini_left = gini_impurity(y_pred_left)
    gini_right = gini_impurity(y_pred_right)

    # Step 4: Compute weighted impunity
    weights_left = (n_asc_cumsum / n_total)
    weights_right = 1 - weights_left
    gini_split = gini_left * weights_left + gini_right * weights_right

    return _best_split_feat(X_[:-1], gini_split)

@njit(cache=True, fastmath=True)
def _best_split_feat(X, gini):
    '''Proposes a split based on a sorted vector X and vector of weighted gini impunities
    
    Args:
        X (Array): A 1-d array of X values
        gini (Array): A 1-d array of weighted gini impunities assuming split occurs at X[idx]
        
    Returns:
        A tuple of the split value for X and it's associated gini impunity
    '''
    idx_split = np.argmin(gini)
    # Taking mean of split values to align with sklearn
    x_split = X[idx_split:idx_split+2].mean()
    gini_split = gini[idx_split]

    return (x_split, gini_split)