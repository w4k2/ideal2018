"""
Method.

Hamming and others.
"""
import numpy as np
import helper as h
from sklearn import metrics


class FeatureSelectingEnsemble():
    """
    Feature Selecting Ensemble.

    Base structure for genetic algorithm.
    """

    def __init__(self, base_clf, alpha=.5, beta=.5, n_members=5,
                 p=.5, parents=None):
        """
        Initializer.

        p - probability of using a feature
        """
        self.alpha = alpha
        self.beta = beta
        self.n_members = n_members
        self.p = p
        self.parents = parents
        self.base_clf = base_clf

    def fit(self, X, y):
        """Compute model."""
        self.X = X
        self.y = y
        self.d = X.shape[1]
        if not hasattr(self, 'selected_features'):
            # Random features if none established
            self.selected_features = np.zeros((self.n_members,
                                               self.d)).astype(bool)
            self.selected_features = np.random.choice(a=[False, True],
                                                      size=(self.n_members,
                                                            self.d),
                                                      p=[1-self.p, self.p])
        # Establishing ensemble
        self.ensemble = []
        for features_mask in self.selected_features:
            X_ = self.X[:, features_mask]
            clf = self.base_clf()
            clf.fit(X_, y)
            self.ensemble.append(clf)

    def bac(self, X, y):
        """Balanced accuracy score."""
        # Calculate ensemble support matrix
        esm = []
        for i, clf in enumerate(self.ensemble):
            sv = clf.predict_proba(X[:, self.selected_features[i]])
            esm.append(sv)
        esm = np.array(esm)

        # Calculate support as mean for member classifiers
        mesm = np.mean(esm, axis=0)

        # Prediction and score
        prediction = np.argmax(mesm, axis=1)
        score = metrics.balanced_accuracy_score(y, prediction)

        return score

    def quality(self, X, y):
        """Optimization criteria."""
        a = self.alpha * self.features_proportion()
        b = self.beta * self.average_hamming()
        return self.bac(X, y) - a + b

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
