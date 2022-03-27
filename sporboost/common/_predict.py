from numba import njit
import numpy as np

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
