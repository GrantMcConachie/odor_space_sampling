"""
Plotting functions
"""

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
