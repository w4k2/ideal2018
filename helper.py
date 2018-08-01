"""Universal helper for W4K2."""
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
ds_dir = "datasets"
variations = ("bare", "e_r", "e_w", "e_n", "s_r", "s_w", "s_n")


def plot(dataset, alphas, betas, sumtable_scores, sumtable_winners, bs):
    """Plot whole search result."""
    cmap = plt.cm.Blues
    cmap = plt.cm.coolwarm
    maxs = np.max(sumtable_scores)
    diff = np.abs(maxs - bs)

    plt.imshow(sumtable_scores, interpolation='nearest',
               cmap=cmap, vmin=bs - diff, vmax=bs + diff)
    plt.title("%s (bare %.3f)" % (dataset, bs))
    plt.colorbar()

    plt.yticks(np.arange(len(alphas)),
               alphas, rotation=45)
    plt.xticks(np.arange(len(betas)),
               betas, rotation=45)

    tresh = .5
    if sumtable_winners is not None:
        for i, j in itertools.product(range(len(alphas)),
                                      range(len(betas))):
            plt.text(j, i, "%s\n%.3f" % (variations[sumtable_winners[i, j]],
                                         sumtable_scores[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if sumtable_scores[i, j] > tresh else "black")
    else:
        for i, j in itertools.product(range(len(alphas)),
                                      range(len(betas))):
            plt.text(j, i, "%.3f" % (sumtable_scores[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if sumtable_scores[i, j] > tresh else "black")

    plt.tight_layout()
    plt.ylabel('alpha')
    plt.xlabel('beta')

    plt.savefig("plots/%s.png" % dataset)
    plt.clf()


def hamming(a, b):
    """Calculate hamming measure for two logic vectors."""
    return np.sum(a ^ b) / a.shape[0]


def load_keel(string, separator=","):
    """Load arff file from keel dataset."""
    f = open(string, "r")
    s = [line for line in f]
    f.close()

    s = filter(lambda e: e[0] != '@', s)
    s = [v.strip().split(separator) for v in s]
    df = np.array(s)
    X = np.asarray(df[:, :-1], dtype=float)
    d = {'positive': 1, 'negative': 0}
    y = np.asarray([d[v[-1].strip()] if v[-1].strip() in d
                    else v[-1].strip() for v in s])

    y = y.astype(int)

    return X, y


def load_dataset(dataset):
    """Load given dataset."""
    # Load full dataset
    X, y = load_keel("%s/%s/%s.dat" % (
        ds_dir, dataset, dataset
    ))
    X_, y_ = [], []
    # Load and process folds
    for i in range(1, 6):
        try:
            X_train, y_train = load_keel("%s/%s/%s-5-fold/%s-5-%itra.dat" % (
                ds_dir, dataset, dataset, dataset, i
            ))
            X_test, y_test = load_keel("%s/%s/%s-5-fold/%s-5-%itst.dat" % (
                ds_dir, dataset, dataset, dataset, i
            ))
        except FileNotFoundError:
            X_train, y_train = load_keel("%s/%s/%s-5-%itra.dat" % (
                ds_dir, dataset, dataset, i
            ))
            X_test, y_test = load_keel("%s/%s/%s-5-%itst.dat" % (
                ds_dir, dataset, dataset, i
            ))
        X_.append((X_train, X_test))
        y_.append((y_train, y_test))
    return (X, y, X_, y_)


def datasets():
    """Return datasets list."""
    datasets = []
    ds_list = sorted(os.listdir('datasets'))
    for ds_idx, ds_name in enumerate(ds_list):
        if ds_name[0] == '.' or ds_name[0] == '_':
            continue
        datasets.append(ds_name)
    return datasets
