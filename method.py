"""
Method.

Hamming and others.
"""
import numpy as np
import helper as h
from sklearn import metrics


class GeneticFeatureSelectionEnsemble():
    """
    GFSE.

    Lorem ipsum dolor sit amet.
    """

    def __init__(self, base_clf, n_candidates=20, a=.5, b=.5):
        """Init."""
        self.base_clf = base_clf
        self.n_candidates = n_candidates
        self.a = a
        self.b = b

    def fit(self, X, y):
        """Compute models."""
        self.candidates = []
        for i in range(self.n_candidates):
            candidate = RandomFeatureEnsemble(self.base_clf, alpha=self.a,
                                              beta=self.b)
            candidate.fit(X, y)
            self.candidates.append(candidate)

        scores = np.array([c.quality(X, y) for c in self.candidates])
        self.qualities = scores[:, 0]
        self.bacs = scores[:, 1]

        places = np.argsort(1-self.qualities)

        # print(places)
        # print(self.qualities)

        self.qualities = self.qualities[places]
        self.bacs = self.bacs[places]
        self.candidates = [self.candidates[i] for i in places]

        # print(self.qualities)

    def bac(self, X, y):
        """Balanced accuracy score."""
        return self.candidates[0].bac(X, y)


class RandomFeatureEnsemble():
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

        # Prevent for empty features set
        for row in self.selected_features:
            if np.sum(row) == 0:
                row[np.random.randint(self.d)] = True

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
        bac = self.bac(X, y)
        a = self.alpha * self.features_proportion()
        b = self.beta * self.average_hamming()
        return bac - a + b, bac

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
