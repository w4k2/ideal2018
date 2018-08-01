"""
IDEAL2018 experiments.

Multi-class imbalanced data classification based on feature selection
techniques.
"""
import helper as h

title = "Multi-class imbalanced data classification based on feature selection techniques"

abstract = "The difficulty of the many practical decision problem lies in the nature of analyzed data. One of the most important real data characteristic is imbalance among examples from different classes. Despite more than two decades of research, imbalanced data classification is still one of the vital challenges to be addressed. The traditional classification algorithms display strongly biased performance on imbalanced datasets. In this work a novel classifier ensemble forming technique for multi-class imbalanced datasets is presented, which takes into consideration on the one hand selected features used for training individual classifiers, but on the other hand it ensures an appropriate diversity of a classifier ensemble. The proposed approach has been evaluated on the basis of the computer experiments carried out on the benchmark datasets. Their results seem to confirm the usefulness of the proposed concept in comparison to the state-of-art methods."

print("# %s\n" % title)
print("> %s\n" % abstract)

for dataset in h.datasets():
    print("## %s\n" % dataset)

    print("#### Summary\n")
    print("![](plots/%s.png)\n" % dataset)

    for i, var in enumerate(h.variations):
        print("#### %s\n" % var)
        print("![](plots/%s_%s.png)\n" % (dataset, var))
