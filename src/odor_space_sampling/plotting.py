"""
Plotting functions
"""

import math
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .data import OdorData
from .utils import get_rd_fun_group_labels, get_ks_stats


def _get_x(data):
    return data.x if isinstance(data, OdorData) else data


def _get_df(data):
    return data.df if isinstance(data, OdorData) else data


def plot_scree_plot(data):
    """
    Generates a scree plot for data

    Args:
        data (OdorData or np.ndarray): odor data object or raw data matrix

    Returns:
        scree plot of data
    """
    x = _get_x(data)
    pca = PCA()
    pca.fit(x)
    plt.scatter(
        range(len(np.cumsum(pca.explained_variance_ratio_))),
        np.cumsum(pca.explained_variance_ratio_),
        s=1.5
    )
    plt.ylabel('Cumulative variance explained')
    plt.xlabel('Number of components')
    plt.show()


def plot_feature_covariance(data):
    """
    Plots a histogram of all the covariances of each feature in data

    Args:
        data (OdorData or np.ndarray): odor data object or raw data matrix

    Returns:
        plot of covariances
    """
    x = _get_x(data)
    cov = np.corrcoef(x.T)
    up_tri_ind = np.triu_indices(x.shape[1])
    plt.hist(cov[up_tri_ind])
    plt.xlabel('covariance')
    plt.ylabel('count')
    plt.title('covariance of all features in odor space')


def plot_fun_group_dist(data, label="Label", density=True, ylim=None):
    """
    Plots normalized functional group distributions as a grouped bar chart.

    Args:
        data (OdorData or list[tuple]): one of:
            - a single OdorData object
            - a list of (OdorData, label) pairs for grouped bars
        label (str): label for the bars when a single OdorData is passed
        density (bool): whether to plot density or counts
        ylim (list): axis limit for the yaxis

    Returns:
        None (displays plot)

    Examples:
        # single dataset
        plot_fun_group_dist(data, label="All odors")

        # multiple datasets side-by-side
        plot_fun_group_dist([
            (data_all,   "All"),
            (data_human, "Human"),
            (data_gmm,   "GMM"),
        ])
    """
    # normalize to list of (OdorData, label) pairs
    if isinstance(data, OdorData):
        datasets = [(data, label)]
    else:
        datasets = data

    # compute normalized functional group counts for each dataset
    all_counts = []
    desc_name = None
    for od, lbl in datasets:
        smiles_list = list(od.df['smiles'])
        freq, names = get_rd_fun_group_labels(smiles_list)
        freq = np.array(freq)
        counts = np.nansum(freq, axis=0).astype(float)
        if density:
            counts /= counts.sum()
        all_counts.append((counts, lbl))
        if desc_name is None:
            desc_name = names

    # grouped bar chart
    n = len(all_counts)
    bar_width = 0.8 / n
    x = np.arange(len(desc_name))

    fig, ax = plt.subplots(figsize=(20, 6))
    for i, (counts, lbl) in enumerate(all_counts):
        offset = (i - n / 2 + 0.5) * bar_width
        ax.bar(x + offset, counts, bar_width, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(desc_name, rotation=90, ha='center', fontsize=8)
    ax.set_xlim(-0.5, len(desc_name) - 0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel('density')
    ax.set_title('Functional Group Distribution')
    if any(lbl for _, lbl in all_counts):
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ks_dist(reference, test_methods, label="Test"):
    """
    Prints KS statistic summary stats and plots their distributions for one or
    more test datasets compared against a reference dataset.

    Args:
        reference (OdorData): the reference dataset to compare against
            (e.g. the full odor space)
        test_methods (OdorData or list[tuple]): one of:
            - a single OdorData object
            - a list of (OdorData, label) pairs
        label (str): label to use when a single OdorData is passed directly

    Returns:
        ks_stats (dict): {label: list of KS statistic values} for each test

    Examples:
        # one test dataset
        plot_ks_dist(data_all, data_gmm, label="GMM")

        # multiple test datasets
        plot_ks_dist(data_all, [
            (data_gmm,    "GMM"),
            (data_kmeans, "KMeans"),
            (data_uniform, "Uniform"),
        ])
    """
    if isinstance(test_methods, OdorData):
        test_methods = [(test_methods, label)]

    # compute KS stats for each test dataset vs reference
    ks_stats = {}
    for od, lbl in test_methods:
        ks_values, _, _ = get_ks_stats(reference.df, od.df)
        stats = [v.statistic for v in ks_values]
        ks_stats[lbl] = stats
        print(lbl)
        print(f'  mean:   {np.nanmean(stats):.4f}')
        print(f'  median: {np.nanmedian(stats):.4f}')

    # plot histograms in a grid
    keys = list(ks_stats.keys())
    n = len(keys)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axs = axs.flatten()

    for i, key in enumerate(keys):
        axs[i].hist(ks_stats[key], alpha=0.6, bins=30)
        axs[i].set_title(key)
        axs[i].set_xlabel('KS statistic')
        axs[i].set_ylabel('count')

    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    return ks_stats


def plot_ecdf_of_feature():
    # TODO: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html
    pass


def plot_sampling_projections(data, sample_methods, label="Samples", umap_seed=42):
    """
    Plots each set of sampled indices in PCA and UMAP 2D space.

    Args:
        data (OdorData or np.ndarray): odor data object or raw data matrix
        sample_methods: one of:
            - a (samples, indices, distances) tuple from a single sampling function
            - a plain indices array
            - a list of (result_or_indices, label) pairs, where each result_or_indices
              is either a (samples, indices, distances) tuple or a plain indices array
        label (str): label to use when a single result or indices array is passed
            directly (default "Samples")
        umap_seed (int): random seed for UMAP (default 42)

    Returns:
        None (displays plots)

    Examples:
        # single sampling function result
        result = sampling.gaussian_sample(data, n_samples=20, seed=42)
        plot_sampling_projections(data, result, label="Gaussian")

        # plain indices
        plot_sampling_projections(data, my_indices, label="TORI")

        # multiple methods
        plot_sampling_projections(data, [(result, "Gaussian"), (my_indices, "TORI")])
    """
    x = _get_x(data)

    # normalize single inputs into list format
    if isinstance(sample_methods, np.ndarray):
        sample_methods = [(sample_methods, label)]
    elif isinstance(sample_methods, tuple) and len(sample_methods) == 3:  # if just one sample is input into the function
        sample_methods = [(sample_methods, label)]

    pca_vis = PCA(n_components=2)
    x_pca2 = pca_vis.fit_transform(x)

    reducer = umap.UMAP(n_components=2, random_state=umap_seed)
    embedding = reducer.fit_transform(x)

    for item, lbl in sample_methods:
        indices = item[1] if isinstance(item, tuple) else item

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].scatter(x_pca2[:, 0], x_pca2[:, 1], c='grey', s=1, alpha=0.2, label='All Data')
        axes[0].scatter(x_pca2[indices, 0], x_pca2[indices, 1], c='red', s=10, label=f'{lbl} Samples')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title(f'{lbl} Sample (PCA space)')
        axes[0].legend()

        axes[1].scatter(embedding[:, 0], embedding[:, 1], c='grey', s=1, alpha=0.2, label='All Data')
        axes[1].scatter(embedding[indices, 0], embedding[indices, 1], c='red', s=10, label=f'{lbl} Samples')
        axes[1].set_xlabel('UMAP1')
        axes[1].set_ylabel('UMAP2')
        axes[1].set_title(f'{lbl} Sample (UMAP space)')
        axes[1].legend()

        plt.tight_layout()

    plt.show()


def plot_all_sampling_method_points(data, results, extra_methods=None):
    """
    Convenience wrapper around plot_sampling_projections for the dict output
    of sample_with_all_methods.

    Args:
        data (OdorData or np.ndarray): odor data object or raw data matrix
        results (dict): output from sample_with_all_methods — keys are method
            names, values are dicts with "samples", "indices", and "distances" keys
        extra_methods (list[tuple], optional): additional ((samples, indices, distances), label)
            pairs, e.g. from individual sampling functions or external index sets

    Returns:
        None (displays plots)
    """
    sample_methods = [
        ((v["samples"], v["indices"], v["distances"]), k) for k, v in results.items()
    ]
    if extra_methods:
        sample_methods.extend(extra_methods)
    plot_sampling_projections(data, sample_methods)


def plot_coverage(data, sample_methods):
    """
    Plots a bar chart of mean nearest-neighbour distance (coverage) for each
    sampling method. Lower mean distance = better coverage of the space.

    Args:
        data (OdorData or np.ndarray): odor data object or raw data matrix
        sample_methods (list[tuple]): list of (indices, label) pairs
    """
    x = _get_x(data)
    labels, means, stds = [], [], []

    for indices, lbl in sample_methods:
        D = pairwise_distances(x, x[indices])
        min_dists = np.min(D, axis=1)
        means.append(np.mean(min_dists))
        stds.append(np.std(min_dists))
        labels.append(lbl)
        print(f'{lbl}: mean min-dist = {np.mean(min_dists):.4f}, std = {np.std(min_dists):.4f}')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_ylabel('Mean nearest distance')
    ax.set_title('Coverage by Sampling Method')
    plt.tight_layout()
    plt.show()


def plot_all_sampling_methods_coverage(data, results, extra_methods=None):
    """
    Plots coverage statistics for all sampling methods in results.

    Args:
        data (OdorData or np.ndarray): odor data object or raw data matrix
        results (dict): output from sample_with_all_methods
        extra_methods (list[tuple], optional): additional (indices, label) pairs
    """
    sample_methods = [(v["indices"], k) for k, v in results.items()]
    if extra_methods:
        sample_methods.extend(extra_methods)
    plot_coverage(data, sample_methods)


def plot_fn_groups(data, sample_methods, save_path=None):
    """
    Plots the number of unique functional groups covered by each sampling
    method as a bar chart.

    Args:
        data (OdorData): odor data object
        sample_methods (list[tuple]): list of (indices, label) pairs
        save_path (str, optional): if provided, saves missing functional groups
            per method to a txt file at this path
    """
    df = _get_df(data)
    labels, num_fr = [], []
    missing_groups = {}
    desc_names = None

    for indices, lbl in sample_methods:
        chosen_smiles = list(df.iloc[indices]['smiles'])
        rd_desc, names = get_rd_fun_group_labels(chosen_smiles)
        if desc_names is None:
            desc_names = names
        present = np.array(rd_desc).sum(axis=0) > 0
        num_fr.append(int(present.sum()))
        labels.append(lbl)
        missing_groups[lbl] = [g for g, p in zip(names, present) if not p]

    if save_path is not None:
        with open(save_path, 'w') as f:
            for lbl, missing in missing_groups.items():
                f.write(f'Missing functional groups in {lbl}:\n')
                f.write('\n'.join(missing))
                f.write('\n\n')

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, num_fr)
    ax.bar_label(bars)
    ax.set_ylabel('count')
    ax.set_title(f'Number of functional groups sampled (out of {len(desc_names)})')
    plt.tight_layout()
    plt.show()


def plot_all_sampling_methods_fun_groups(data, results, extra_methods=None, save_path=None):
    """
    Plots functional group coverage for all sampling methods in results.

    Args:
        data (OdorData): odor data object
        results (dict): output from sample_with_all_methods
        extra_methods (list[tuple], optional): additional (indices, label) pairs
        save_path (str, optional): if provided, saves missing functional groups
            per method to a txt file at this path
    """
    sample_methods = [(v["indices"], k) for k, v in results.items()]
    if extra_methods:
        sample_methods.extend(extra_methods)
    plot_fn_groups(data, sample_methods, save_path=save_path)


def plot_data_dist(data, sample_methods, density=False):
    """
    Plots the distribution of data types (human / gslf / both) for each
    sampling method as a grouped bar chart.

    Args:
        data (OdorData): odor data object
        sample_methods (list[tuple]): list of (indices, label) pairs
        density (bool): if True, normalize counts to fractions (default False)
    """
    df = _get_df(data)
    categories, num_hum, num_gslf, num_both = [], [], [], []

    for indices, lbl in sample_methods:
        hum = gslf = both = 0
        for l in df.iloc[indices]['label']:
            if 'human' in l and 'gslf' in l:
                both += 1
            elif 'human' in l:
                hum += 1
            elif 'gslf' in l:
                gslf += 1

        if density:
            total = hum + gslf + both
            if total > 0:
                hum, gslf, both = hum / total, gslf / total, both / total

        categories.append(lbl)
        num_hum.append(hum)
        num_gslf.append(gslf)
        num_both.append(both)

    x = np.arange(len(categories))
    bar_width = 0.25
    fmt = '%.2f' if density else '%d'

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - bar_width, num_gslf, bar_width, label='gslf')
    bar2 = ax.bar(x,             num_hum,  bar_width, label='human')
    bar3 = ax.bar(x + bar_width, num_both, bar_width, label='both')
    ax.bar_label(bar1, fmt=fmt)
    ax.bar_label(bar2, fmt=fmt)
    ax.bar_label(bar3, fmt=fmt)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('density' if density else 'count')
    ax.set_title('Sampled dataset distribution')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_all_sampling_methods_data_dist(data, results, extra_methods=None, density=False):
    """
    Plots data type distribution for all sampling methods in results.

    Args:
        data (OdorData): odor data object
        results (dict): output from sample_with_all_methods
        extra_methods (list[tuple], optional): additional (indices, label) pairs
        density (bool): if True, normalize counts to fractions (default False)
    """
    sample_methods = [(v["indices"], k) for k, v in results.items()]
    if extra_methods:
        sample_methods.extend(extra_methods)
    plot_data_dist(data, sample_methods, density=density)


def plot_gmm_sweep(aics, bics, ks_means, ks_meds):
    """
    plotting aic and bic figures for gmm sweep

    Args:
        aics (list): Akaike information criterion values for each fitted GMM
        bics (list): bayesian information criterion values for each fitted GMM
        ks_means (list): mean Kolmogorov-Smirnov (K-S) statistic between
                         sampled points and full data distribution for all data
                         features
        ks_meds (list): median Kolmogorov-Smirnov (K-S) statistic between
                        sampled points and full data distribution for all data
                        features

    Returns:
        plot of all args
    """
    for i, name in zip(
        [aics, bics, ks_means, ks_meds],
        ["AIC", "BIC", "Mean KS", "Median KS"]
    ):
        plt.figure()
        plt.style.use("seaborn-v0_8-whitegrid")  # nice default style
        x = np.arange(1, len(i) + 1)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

        ax.plot(
            x, i,
            marker='o', markersize=4,
            linewidth=1.5,
            color='#1f77b4',
            label=name
        )

        # Labels and title
        ax.set_xlabel('Number of clusters', fontsize=11)
        ax.set_ylabel(f'{name}', fontsize=11)
        ax.set_title(f'{name} vs Number of Clusters', fontsize=12, pad=8)
        ax.margins(x=0.02, y=0.05)
        min_idx = np.argmin(i)
        ax.scatter(
            x[min_idx],
            i[min_idx],
            color='crimson',
            s=30,
            zorder=3,
            label=f'Minimum {name}'
        )
        ax.legend(frameon=False, fontsize=9)

        plt.tight_layout()
    plt.show()
