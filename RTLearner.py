import numpy as np


def _mode_1d(arr):
    """Mode of a 1D array using numpy (scipy-version-independent)."""
    vals, counts = np.unique(arr, return_counts=True)
    return float(vals[np.argmax(counts)])


class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def add_evidence(self, data_x, data_y):
        self.tree = self._build_tree(data_x, data_y)
        if self.verbose:
            print(f"Tree shape: {self.tree.shape}")

    def query(self, points):
        y_pred = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            y_pred[i] = self._predict(points[i, :], 0)
        return y_pred

    def _predict(self, point, node):
        node = int(node)
        if self.tree[node, 0] == -1:
            return self.tree[node, 1]
        feature = int(self.tree[node, 0])
        if point[feature] <= self.tree[node, 1]:
            return self._predict(point, node + self.tree[node, 2])
        else:
            return self._predict(point, node + self.tree[node, 3])

    def _build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size or data_x.shape[0] == 0:
            leaf_val = _mode_1d(data_y) if data_y.size > 0 else 0.0
            return np.array([[-1, leaf_val, np.nan, np.nan]])

        if np.all(data_y == data_y[0]):
            return np.array([[-1, float(data_y[0]), np.nan, np.nan]])

        # Randomly select a feature to split on (all features eligible)
        rand_feature = np.random.randint(0, data_x.shape[1])
        split_val = np.median(data_x[:, rand_feature])

        left_mask = data_x[:, rand_feature] <= split_val
        right_mask = ~left_mask

        # Guard against infinite recursion when median doesn't split the data
        if np.all(left_mask) or not np.any(left_mask):
            split_val = np.mean(data_x[:, rand_feature])
            left_mask = data_x[:, rand_feature] <= split_val
            right_mask = ~left_mask

        # If still can't split, return a leaf
        if np.all(left_mask) or not np.any(left_mask):
            return np.array([[-1, _mode_1d(data_y), np.nan, np.nan]])

        left_tree = self._build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self._build_tree(data_x[right_mask], data_y[right_mask])

        root = np.array([[rand_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))
