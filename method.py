"""
Method.

Hamming and others.
"""
import numpy as np
import helper as h
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score as bas


class FeatureSelectionEnsemble():
    """
    FSE.

    Lorem ipsum dolor sit amet.
    """

    def __init__(self, base_clf, n_candidates=20, n_members=5,
                 alpha=.5, beta=.5, p=.5):
        """Init."""
        self.base_clf = base_clf
        self.n_candidates = n_candidates
        self.n_members = n_members
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def fit(self, X, y):
        """Compute models."""
        self.candidates = []
        for i in range(self.n_candidates):
            candidate = RandomFeatureEnsemble(self.base_clf, p=self.p,
                                              alpha=self.alpha,
                                              beta=self.beta,
                                              n_members=self.n_members)
            candidate.fit(X, y)
            self.candidates.append(candidate)

    def bac(self, X, y, weighting):
        """Balanced accuracy score."""
        scores = np.array([c.quality(X, y, weighting)
                           for c in self.candidates])
        self.qualities = scores[:, 0]
        self.bacs = scores[:, 1]

        places = np.argsort(1-self.qualities)

        self.qualities = self.qualities[places]
        self.bacs = self.bacs[places]
        self.candidates = [self.candidates[i] for i in places]

        return self.candidates[0].bac(X, y, weighting)


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
        self.weights = []
        for features_mask in self.selected_features:
            X_ = self.X[:, features_mask]
            clf = self.base_clf()
            clf.fit(X_, y)
            self.ensemble.append(clf)
            self.weights.append(bas(y, clf.predict(X[:, features_mask])))
        self.weights = np.array(self.weights)

        scaler = preprocessing.MinMaxScaler()
        self.nweights = scaler.fit_transform(self.weights.reshape(-1, 1)).T[0]
        self.nweights += .01

    def bac(self, X, y, weighting):
        """Balanced accuracy score."""
        # Calculate ensemble support matrix
        esm = []
        for i, clf in enumerate(self.ensemble):
            sv = clf.predict_proba(X[:, self.selected_features[i]])
            esm.append(sv)
        esm = np.array(esm)

        # Calculate support as mean for member classifiers
        if weighting == 1:
            esm *= self.weights[:, np.newaxis, np.newaxis]
        elif weighting == 2:
            esm *= self.nweights[:, np.newaxis, np.newaxis]

        mesm = np.mean(esm, axis=0)

        # Prediction and score
        prediction = np.argmax(mesm, axis=1)
        score = bas(y, prediction)

        return score

    def quality(self, X, y, weighting):
        """Optimization criteria."""
        bac = self.bac(X, y, weighting)
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
