import numpy as np


def _col_mode(Y):
    """Mode along axis=0 for a 2D array (bags x samples). Returns shape (n_samples,)."""
    result = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        vals, counts = np.unique(Y[:, j], return_counts=True)
        result[j] = vals[np.argmax(counts)]
    return result


class BagLearner:
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [learner(**kwargs) for _ in range(bags)]

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            bag_idx = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner.add_evidence(data_x[bag_idx, :], data_y[bag_idx])

    def query(self, points):
        Y = np.zeros((self.bags, points.shape[0]))
        for i, learner in enumerate(self.learners):
            Y[i, :] = learner.query(points)
        # Return shape (1, n_points) so callers can index as pred_y[0, i]
        return _col_mode(Y).reshape(1, -1)
