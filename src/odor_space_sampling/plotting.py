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


def plot_sampling_projections(x, sample_methods):
    """
    Plots each set of sampled indices in PCA and UMAP 2D space.

    Args:
        x (np.ndarray): full data matrix (n_samples, n_features)
        sample_methods (list[tuple]): list of (indices, label) pairs

    Returns:
        None (displays plots)
    """
    pca_vis = PCA(n_components=2)
    x_pca2 = pca_vis.fit_transform(x)

    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(x)

    for indices, label in sample_methods:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].scatter(x_pca2[:, 0], x_pca2[:, 1], c='grey', s=1, alpha=0.2, label='All Data')
        axes[0].scatter(x_pca2[indices, 0], x_pca2[indices, 1], c='red', s=10, label=f'{label} Samples')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title(f'{label} Sampling (PCA space)')
        axes[0].legend()

        axes[1].scatter(embedding[:, 0], embedding[:, 1], c='grey', s=1, alpha=0.2, label='All Data')
        axes[1].scatter(embedding[indices, 0], embedding[indices, 1], c='red', s=10, label=f'{label} Samples')
        axes[1].set_xlabel('UMAP1')
        axes[1].set_ylabel('UMAP2')
        axes[1].set_title(f'{label} Sampling (UMAP space)')
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
            names, values are dicts with an "indices" key
        extra_methods (list[tuple], optional): additional (indices, label) pairs,
            e.g. [(tori_df_1_idx, "TORI_table1"), (tori_df_2_idx, "TORI_table2")]

    Returns:
        None (displays plots)
    """
    sample_methods = [(v["indices"], k) for k, v in results.items()]
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
