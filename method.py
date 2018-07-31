"""
Method.

Hamming and others.
"""
import numpy as np
# import helper as h


class FeatureSelectingEnsemble():
    """
    Feature Selecting Ensemble.

    Base structure for genetic algorithm.
    """

    def __init__(self, alpha=.5, beta=.5, n_members=5):
        """Initializer."""
        self.alpha = alpha
        self.beta = beta
        self.n_members = n_members

    def fit(self, X, y):
        """Compute model."""
        self.d = X.shape[1]
        self.selected_features = np.zeros((self.n_members,
                                           self.d)).astype(bool)
        self.selected_features[0, 0] = 1
        self.selected_features[1, 2] = 1
        self.selected_features[3, 2] = 1
        self.ensemble = []

        # print(self.selected_features)

    def bac(self):
        """Balanced accuracy score."""
        return .5

    def quality(self):
        """Optimization criteria."""
        a = self.alpha * self.features_proportion()
        b = self.beta * self.average_hamming()
        return self.bac() - a - b

    def average_hamming(self):
        """Return average hamming measure between members of commitee."""
        return .5

    def features_proportion(self):
        """Percent of features used in ensemble."""
        v = np.sum(np.sum(self.selected_features, axis=0) > 0)
        return v / self.d
