"""Universal helper for W4K2."""
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from scipy import stats
ds_dir = "datasets"
variations = ("bare", "e_r", "e_w", "e_n", "s_r", "s_w", "s_n")


def analyze(dataset, X, y, res, alphas, betas):
    """Plot whole search result."""
    # Analyze dataset
    row = []
    row.append(dataset.replace('_', '').replace('-',''))

    # IR
    u = np.unique(y)
    first = np.sum(y == u[0])
    second = np.sum(y == u[1])
    print(first, second)
    ir = first/second if first>second else second/first
    row.append("%.0f" % ir)

    # Samples and features
    row.append("%i" % X.shape[0])
    row.append("%i" % X.shape[1])

    # Prepare views for plot
    sumtable_scores = np.zeros((len(alphas), len(betas)))
    sumtable_winners = np.zeros((len(alphas), len(betas))).astype(int)
    bare_score = np.mean(res[0, 0, 0, :])

    # Search for results for parameter
    for a, alpha in enumerate(alphas):
        for b, beta in enumerate(betas):
            local_res = res[a, b, 1:, :]
            mean_scores = np.mean(local_res, axis=1)
            winner = np.argmax(mean_scores) + 1
            score = np.max(mean_scores)

            sumtable_scores[a, b] = score
            sumtable_winners[a, b] = winner

    # Establish best combination
    gs_winner = np.unravel_index(np.argmax(sumtable_scores,
                                           axis=None),
                                 sumtable_scores.shape)

    row.append(("%.1f" % alphas[gs_winner[0]])[1:])
    row.append(("%.1f" % betas[gs_winner[1]])[1:])

    leader = sumtable_winners[gs_winner]
    optimal_scores = res[gs_winner[0], gs_winner[1], :, :]
    print("Overall winner %s [%s]" % (str(gs_winner), variations[leader]))

    optimal_mean = np.mean(optimal_scores, axis=1)
    optimal_std = np.std(optimal_scores, axis=1)
    print(optimal_mean)
    print(optimal_std)

    print(leader)
    if optimal_mean[leader] < optimal_mean[0]:
        leader = 0


    # Analyze dependencies
    optimal_dependencies = []
    a = optimal_scores[leader]
    for i, var in enumerate(variations):
        if i == leader:
            optimal_dependencies.append(True)
        else:
            b = optimal_scores[i]
            if optimal_mean[i] == optimal_mean[leader]:
                is_dependent = True
            else:
                is_dependent = stats.wilcoxon(a, b).pvalue > 0.05
            optimal_dependencies.append(is_dependent)
    optimal_dependencies = np.array(optimal_dependencies)

    for i, od in enumerate(optimal_dependencies):
        if optimal_mean[i] == optimal_mean[leader]:
            row.append("\\cellcolor{green!20} \\underline{%.3f}" % ( optimal_mean[i]))
            row.append("%.3f" % optimal_std[i])
        else:
            row.append("\\cellcolor{%s!20} %.3f" % ("green" if od else "white", optimal_mean[i]))
            row.append("%.3f" % optimal_std[i])

    # Plot barchart
    plt.figure(figsize=(6, 3))
    # fig, ax = plt.subplots()
    plt.title(dataset)
    dep_color = ["green" if d else "red" for d in optimal_dependencies]
    plt.bar(list(variations), optimal_mean, color=dep_color,
            yerr=optimal_std, alpha=.75)

    plt.tight_layout()
    plt.savefig("plots/%s_bar.png" % dataset)
    plt.savefig("plots/%s_bar.eps" % dataset)
    plt.clf()

    # Prepare plot helpers
    cmap = plt.cm.coolwarm
    maxs = np.max(sumtable_scores)
    diff = np.abs(maxs - bare_score)
    vmin = bare_score - diff
    vmax = bare_score + diff

    # Plot summary of GS
    plt.figure(figsize=(5, 6))
    plt.imshow(sumtable_scores, interpolation='nearest', cmap=cmap,
               vmin=vmin, vmax=vmax)
    plt.title("%s (bare %.3f)" % (dataset, bare_score),fontsize=16)
    plt.colorbar(orientation='horizontal',shrink=.86)

    plt.yticks(np.arange(len(alphas)),
               alphas, rotation=0,fontsize=11)
    plt.xticks(np.arange(len(betas)),
               betas, rotation=0,fontsize=11)
    plt.ylabel('alpha', fontsize=11)
    plt.xlabel('beta',fontsize=11)

    for i, j in itertools.product(range(len(alphas)),
                                  range(len(betas))):
        plt.text(j, i, "%s\n%.3f" % (variations[sumtable_winners[i, j]],
                                     sumtable_scores[i, j]),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white",
                 fontsize=12)

    plt.tight_layout()
    plt.savefig("plots/%s.png" % dataset)
    plt.savefig("plots/%s.eps" % dataset)
    plt.clf()

    # Plot summary
    fig, ax = plt.subplots(2, 3, figsize=(6, 4))

    for i in range(1, 7):
        loc_sco = np.mean(res[:, :, i, :], axis=2)
        a, b = (i-1) // 3, (i-1) % 3

        ax[a, b].imshow(loc_sco, cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax[a, b].set_title(variations[i],fontsize=22)
        ax[a, b].set_xticks([])
        ax[a, b].set_yticks([])

    plt.tight_layout()
    plt.savefig("plots/%s_sum.png" % dataset)
    plt.savefig("plots/%s_sum.eps" % dataset)
    plt.clf()

    return row


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
