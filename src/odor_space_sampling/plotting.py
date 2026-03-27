"""
Plotting functions
"""

import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_scree_plot(x):
    """
    Generates a scree plot for data

    Args:
        x (np.ndarray): data to generate a scree plot
    
    Returns:
        scree plot of data
    """
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


def plot_feature_covariance(x):
    """
    Plots a histogram of all the covaraiances of each feature in data

    Args:
        x (np.ndarray): data matrix
    
    Returns:
        plot of covaraiances
    """
    cov = np.corrcoef(x.T)
    up_tri_ind = np.triu_indices(x.shape[1])
    plt.hist(cov[up_tri_ind])
    plt.xlabel('covariance')
    plt.ylabel('count')
    plt.title('covariance of all features in odor space')


def plot_sampling_projections(x, sample_methods, label="Samples", umap_seed=42):
    """
    Plots each set of sampled indices in PCA and UMAP 2D space.

    Args:
        x (np.ndarray): full data matrix (n_samples, n_features)
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
        result = sampling.gaussian_sample(data.x, n_samples=20, seed=42)
        plot_sampling_projections(data.x, result, label="Gaussian")

        # plain indices
        plot_sampling_projections(data.x, my_indices, label="TORI")

        # multiple methods
        plot_sampling_projections(data.x, [(result, "Gaussian"), (my_indices, "TORI")])
    """
    # normalize single inputs into list format
    if isinstance(sample_methods, np.ndarray):
        sample_methods = [(sample_methods, label)]
    elif isinstance(sample_methods, tuple) and len(sample_methods) == 3:
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
        axes[0].set_title(f'{lbl} Sampling (PCA space)')
        axes[0].legend()

        axes[1].scatter(embedding[:, 0], embedding[:, 1], c='grey', s=1, alpha=0.2, label='All Data')
        axes[1].scatter(embedding[indices, 0], embedding[indices, 1], c='red', s=10, label=f'{lbl} Samples')
        axes[1].set_xlabel('UMAP1')
        axes[1].set_ylabel('UMAP2')
        axes[1].set_title(f'{lbl} Sampling (UMAP space)')
        axes[1].legend()

        plt.tight_layout()

    plt.show()


def plot_all_sampling_methods(x, results, extra_methods=None):
    """
    Convenience wrapper around plot_sampling_projections for the dict output
    of sample_with_all_methods.

    Args:
        x (np.ndarray): full data matrix (n_samples, n_features)
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
    plot_sampling_projections(x, sample_methods)


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
