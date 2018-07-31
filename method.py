"""
Method.

Hamming and others.
"""
import numpy as np
import helper as h


class FeatureSelectingEnsemble():
    """
    Feature Selecting Ensemble.

    Base structure for genetic algorithm.
    """

    def __init__(self, alpha=.5, beta=.5, n_members=5, p=.5):
        """
        Initializer.

        p - probability of using a feature
        """
        self.alpha = alpha
        self.beta = beta
        self.n_members = n_members
        self.p = p

    def fit(self, X, y):
        """Compute model."""
        self.d = X.shape[1]
        self.selected_features = np.zeros((self.n_members,
                                           self.d)).astype(bool)
        self.selected_features = np.random.choice(a=[False, True],
                                                  size=(self.n_members,
                                                        self.d),
                                                  p=[1-self.p, self.p])
        self.ensemble = []

    def bac(self):
        """Balanced accuracy score."""
        return .85

    def quality(self):
        """Optimization criteria."""
        a = self.alpha * self.features_proportion()
        b = self.beta * self.average_hamming()
        return self.bac() - a - b

    def average_hamming(self):
        """Return average hamming measure between members of commitee."""
        hammings = []
        for i in range(self.n_members):
            for j in range(i+1, self.n_members):
                hammings.append(h.hamming(self.selected_features[i],
                                          self.selected_features[j]))

        return np.mean(hammings)

    def features_proportion(self):
        """Percent of features used in ensemble."""
        v = np.sum(np.sum(self.selected_features, axis=0) > 0)
        return v / self.d
