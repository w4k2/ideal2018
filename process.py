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

datasets = h.datasets_for_groups([
    "bal",
    "imb_IRlowerThan9",
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2"
])

print('group,dataset,features,bare,one_random_ens,gfse,best_one')

min_features = 8
analyzed_datasets = 0
for dataset in datasets:
    # Load dataset
    X, y, X_, y_ = h.load_dataset(dataset)

    no_classes = len(np.unique(y))
    if no_classes != 2:
        # print("Multiclass -- passing")
        continue

    if X.shape[1] < min_features:
        # print("To few features -- passing")
        continue

    label_corrector = np.max(y) - 1

    bas_bacs = []
    rfe_bacs = []
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

        # One rfe
        rfe = m.RandomFeatureEnsemble(base_clf=base_clf)
        rfe.fit(X_f_train, y_f_train)
        q, bac = rfe.quality(X_f_test, y_f_test)

        # GFSE
        gfse = m.GeneticFeatureSelectionEnsemble(base_clf, a=.1, b=.1)
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
        rfe_bacs.append(bac)
        gfs_bacs.append(gbac)
        bns_bacs.append(np.mean(bens))

    bas_bac = np.mean(bas_bacs)
    rfe_bac = np.mean(rfe_bacs)
    gfs_bac = np.mean(gfs_bacs)
    bns_bac = np.mean(bns_bacs)

    print("%s,%s,%i,%.3f,%.3f,%.3f,%.3f" % (
        dataset[0], dataset[1], X.shape[1], bas_bac, rfe_bac, gfs_bac,
        bns_bac
    ))
    analyzed_datasets += 1

print("%i analyzed datasets" % analyzed_datasets)
