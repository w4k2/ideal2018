"""
IDEAL2018 experiments.

Multi-class imbalanced data classification based on feature selection
techniques.
"""
import numpy as np
import method as m
import helper as h
from sklearn import naive_bayes, metrics

np.random.seed(2)

base_clf = naive_bayes.GaussianNB


def process_instance(dataset, label_corrector, X_, y_, base_clf,
                     alpha=.5, beta=.5,
                     n_candidates=20, n_members=5,
                     p=.5):
    """Process single instance of problem."""
    bas_bacs = []
    gfs_bacs = []
    bns_bacs = []
    for f in range(5):
        # Divide sets
        X_f_train, X_f_test = X_[f][0], X_[f][1]
        y_f_train, y_f_test = (
            y_[f][0] - label_corrector,
            y_[f][1] - label_corrector
        )

        # Bare score
        bare_clf = base_clf()
        bare_clf.fit(X_f_train, y_f_train)
        bare_prediction = bare_clf.predict(X_f_test)
        bare_bac = metrics.balanced_accuracy_score(y_f_test,
                                                   bare_prediction)

        # GFSE
        gfse = m.FeatureSelectionEnsemble(base_clf, alpha=.1, beta=.1)
        gfse.fit(X_f_train, y_f_train)
        gbac = gfse.bac(X_f_test, y_f_test)
        # print(gbac)

        # Best single
        bens = []
        for i, clf in enumerate(gfse.candidates[0].ensemble):
            prediction = clf.predict(
                X_f_test[:,
                         gfse.candidates[0].selected_features[i]]
            )
            a = metrics.balanced_accuracy_score(
                y_f_test,
                prediction
            )
            bens.append(a)

        bas_bacs.append(bare_bac)
        gfs_bacs.append(gbac)
        bns_bacs.append(np.mean(bens))

    bas_bac = np.mean(bas_bacs)
    gfs_bac = np.mean(gfs_bacs)
    bns_bac = np.mean(bns_bacs)

    return (bas_bac, gfs_bac, bns_bac)


print('group,dataset,features,bare,one_random_ens,gfse,best_one')

min_features = 8
analyzed_datasets = 0
for dataset in h.datasets():
    # Load dataset
    X, y, X_, y_ = h.load_dataset(dataset)

    # Analyze dataset
    no_classes = len(np.unique(y))

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
