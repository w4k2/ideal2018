"""
IDEAL2018 experiments.

Multi-class imbalanced data classification based on feature selection
techniques.
"""
import numpy as np
import method as m
import helper as h
from sklearn import naive_bayes
from sklearn.metrics import balanced_accuracy_score as bas
np.set_printoptions(precision=3)

np.random.seed(2)

base_clf = naive_bayes.GaussianNB


def process_instance(dataset, label_corrector, X_, y_, base_clf,
                     alpha=.5, beta=.5,
                     n_candidates=20, n_members=20,
                     p=.5):
    """Process single instance of problem."""
    # Prepare storage for results
    bare_bacs = []
    fse_bacs_0 = []
    fse_bacs_1 = []
    fse_bacs_2 = []
    one_bacs_0 = []
    one_bacs_1 = []
    one_bacs_2 = []

    # Iterate folds
    for f in range(5):
        # Divide sets
        X_f_train, X_f_test = X_[f][0], X_[f][1]
        y_f_train, y_f_test = (
            y_[f][0] - label_corrector,
            y_[f][1] - label_corrector
        )

        # Establish bare score
        bare_clf = base_clf()
        bare_prediction = bare_clf.fit(X_f_train, y_f_train).predict(X_f_test)
        bare_bac = bas(y_f_test, bare_prediction)
        bare_bacs.append(bare_bac)

        # FSE and best member
        fse = m.FeatureSelectionEnsemble(base_clf,
                                         alpha=alpha, beta=beta, p=p,
                                         n_candidates=n_candidates,
                                         n_members=n_members)
        fse.fit(X_f_train, y_f_train)

        # Without weights
        fse_bacs_0.append(fse.bac(X_f_test, y_f_test, weighting=0))
        winner = fse.candidates[0]
        bens = []
        for i, clf in enumerate(winner.ensemble):
            prediction = clf.predict(X_f_test[:, winner.selected_features[i]])
            bens.append(bas(y_f_test, prediction))
        one_bacs_0.append(np.max(bens))

        # Regular weights
        fse_bacs_1.append(fse.bac(X_f_test, y_f_test, weighting=1))
        winner = fse.candidates[0]
        bens = []
        for i, clf in enumerate(winner.ensemble):
            prediction = clf.predict(X_f_test[:, winner.selected_features[i]])
            bens.append(bas(y_f_test, prediction))
        one_bacs_1.append(np.max(bens))

        # Normalized weights
        fse_bacs_2.append(fse.bac(X_f_test, y_f_test, weighting=2))
        winner = fse.candidates[0]
        bens = []
        for i, clf in enumerate(winner.ensemble):
            prediction = clf.predict(X_f_test[:, winner.selected_features[i]])
            bens.append(bas(y_f_test, prediction))
        one_bacs_2.append(np.max(bens))

    bare_bac = np.mean(bare_bacs)
    fse_bac_0 = np.mean(fse_bacs_0)
    fse_bac_1 = np.mean(fse_bacs_1)
    fse_bac_2 = np.mean(fse_bacs_2)
    one_bac_0 = np.mean(one_bacs_0)
    one_bac_1 = np.mean(one_bacs_1)
    one_bac_2 = np.mean(one_bacs_2)

    return (bare_bac,
            fse_bac_0, fse_bac_1, fse_bac_2,
            one_bac_0, one_bac_1, one_bac_2,
            np.argmax([bare_bac,
                       fse_bac_0, fse_bac_1, fse_bac_2,
                       one_bac_0, one_bac_1, one_bac_2])
            )


min_features = 8
analyzed_datasets = 0
for dataset in h.datasets():
    # Load dataset
    X, y, X_, y_ = h.load_dataset(dataset)

    # Analyze dataset
    no_classes = len(np.unique(y))
    print(dataset)

    # Process instances of problem
    label_corrector = np.max(y) - 1
    res = process_instance(
        dataset, label_corrector=label_corrector,
        X_=X_, y_=y_, base_clf=base_clf,
        alpha=.5, beta=.5, p=.5,
        n_candidates=20, n_members=5
    )
    print(np.array(res))

    analyzed_datasets += 1
