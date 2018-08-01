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

alphas = np.array([0, .1, .3, .5, .7, .9, 1])
betas = np.array([0, .1, .3, .5, .7, .9, 1])

alphas = np.array([0, .25, .5, .75, 1])
betas = np.array([0, .25, .5, .75, 1])

np.random.seed(2)

base_clf = naive_bayes.GaussianNB


def process_instance(dataset, label_corrector, X_, y_, base_clf,
                     n_candidates=20, n_members=20,
                     p=.5):
    """Process single instance of problem."""
    # Alpha, beta, variation, fold
    results = np.zeros((len(alphas), len(betas), 7, 5))

    # Prepare storage for results
    bare_bacs = np.zeros(5)
    fse_bacs_0 = np.zeros((len(alphas), len(betas), 5))
    fse_bacs_1 = np.zeros((len(alphas), len(betas), 5))
    fse_bacs_2 = np.zeros((len(alphas), len(betas), 5))
    one_bacs_0 = np.zeros((len(alphas), len(betas), 5))
    one_bacs_1 = np.zeros((len(alphas), len(betas), 5))
    one_bacs_2 = np.zeros((len(alphas), len(betas), 5))

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
        bare_prediction = bare_clf.fit(X_f_train,
                                       y_f_train).predict(X_f_test)
        bare_bac = bas(y_f_test, bare_prediction)
        bare_bacs[f] = bare_bac

        # FSE and best member
        fse = m.FeatureSelectionEnsemble(base_clf, p=p,
                                         n_candidates=n_candidates,
                                         n_members=n_members)
        fse.fit(X_f_train, y_f_train)

        for a, alpha in enumerate(alphas):
            for b, beta in enumerate(betas):
                # Without weights
                fse_bacs_0[a, b, f] = fse.bac(X_f_test, y_f_test,
                                              alpha, beta, weighting=0)
                winner = fse.candidates[0]
                bens = []
                for i, clf in enumerate(winner.ensemble):
                    prediction = clf.predict(X_f_test[:, winner.selected_features[i]])
                    bens.append(bas(y_f_test, prediction))
                one_bacs_0[a, b, f] = np.max(bens)

                # Regular weights
                fse_bacs_1[a, b, f] = fse.bac(X_f_test, y_f_test,
                                              alpha, beta, weighting=1)
                winner = fse.candidates[0]
                bens = []
                for i, clf in enumerate(winner.ensemble):
                    prediction = clf.predict(X_f_test[:, winner.selected_features[i]])
                    bens.append(bas(y_f_test, prediction))
                one_bacs_1[a, b, f] = np.max(bens)

                # Normalized weights
                fse_bacs_2[a, b, f] = fse.bac(X_f_test, y_f_test,
                                              alpha, beta, weighting=2)
                winner = fse.candidates[0]
                bens = []
                for i, clf in enumerate(winner.ensemble):
                    prediction = clf.predict(X_f_test[:, winner.selected_features[i]])
                    bens.append(bas(y_f_test, prediction))
                one_bacs_2[a, b, f] = np.max(bens)

    results[:, :, 0, :] = bare_bacs
    results[:, :, 1, :] = fse_bacs_0
    results[:, :, 2, :] = fse_bacs_1
    results[:, :, 3, :] = fse_bacs_2
    results[:, :, 4, :] = one_bacs_0
    results[:, :, 5, :] = one_bacs_1
    results[:, :, 6, :] = one_bacs_2

    return results


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
        X_=X_, y_=y_, base_clf=base_clf, p=.5,
        n_candidates=20, n_members=5
    )

    sumtable_scores = np.zeros((len(alphas), len(betas)))
    sumtable_winners = np.zeros((len(alphas), len(betas))).astype(int)

    for a, alpha in enumerate(alphas):
        row_scores = []
        row_winners = []
        for b, beta in enumerate(betas):
            local_res = res[a, b, 1:, :]
            mean_scores = np.mean(local_res, axis=1)
            bare_score = np.mean(res[a, b, 0, :])
            winner = np.argmax(mean_scores)
            score = np.max(mean_scores)

            sumtable_scores[a, b] = score
            sumtable_winners[a, b] = winner

    h.plot(dataset, alphas, betas, sumtable_scores, sumtable_winners,
           bare_score)

    for i, var in enumerate(h.variations):
        loc_sco = np.mean(res[:, :, i+1, :], axis=2)
        h.plot("%s_%s" % (dataset, var), alphas, betas,
               loc_sco, None, bare_score)

    analyzed_datasets += 1
