"""
IDEAL2018 experiments.

Multi-class imbalanced data classification based on feature selection
techniques.
"""
# import numpy as np
import method as m
import helper as h
from sklearn import svm

datasets = h.datasets_for_groups([
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2"
])

for dataset in datasets:
    print(dataset)
    # Load dataset
    X, y, X_, y_ = h.load_dataset(dataset)
    print(X.shape)

    fse = m.FeatureSelectingEnsemble()
    fse.fit(X, y)

    print(fse.quality())
    exit()

base_clf = svm.SVC()
