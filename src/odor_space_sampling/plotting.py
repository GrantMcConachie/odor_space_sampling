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
