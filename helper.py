"""Universal helper for W4K2."""
import numpy as np
import os
ds_dir = "datasets"


def hamming(a, b):
    """Calculate hamming measure for two logic vectors."""
    return np.sum(a ^ b) / a.shape[0]


def load_keel(string, separator=","):
    """Load arff file from keel dataset."""
    try:
        f = open(string, "r")
        s = [line for line in f]
        f.close()
    except:
        raise Exception

    s = filter(lambda e: e[0] != '@', s)
    s = [v.strip().split(separator) for v in s]
    df = np.array(s)
    X = np.asarray(df[:, :-1], dtype=float)
    d = {'positive': 1, 'negative': 0}
    y = np.asarray([d[v[-1].strip()] if v[-1].strip() in d
                    else v[-1].strip() for v in s])

    return X, y


def load_dataset(dataset):
    """Load given dataset."""
    group_path, ds_name = dataset
    # Load full dataset
    X, y = load_keel("%s/%s/%s.dat" % (
        group_path, ds_name, ds_name
    ))
    X_, y_ = [], []
    # Load and process folds
    for i in range(1, 6):
        X_train, y_train = load_keel("%s/%s/%s-5-fold/%s-5-%itra.dat" % (
            group_path, ds_name, ds_name, ds_name, i
        ))
        X_test, y_test = load_keel("%s/%s/%s-5-fold/%s-5-%itst.dat" % (
            group_path, ds_name, ds_name, ds_name, i
        ))
        X_.append((X_train, X_test))
        y_.append((y_train, y_test))
    return (X, y, X_, y_)


def datasets_for_groups(ds_groups):
    """Return datasets for a tuple of groups."""
    datasets = []
    for group_idx, ds_group in enumerate(ds_groups):
        group_path = "%s/%s" % (ds_dir, ds_group)
        ds_list = sorted(os.listdir(group_path))
        for ds_idx, ds_name in enumerate(ds_list):
            if ds_name[0] == '.' or ds_name[0] == '_':
                continue
            datasets.append((group_path, ds_name))
    return datasets
