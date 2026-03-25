"""
functions for sampling odor space
"""

import numpy as np
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min


def uniform_sample(x, n_samples, seed):
    rng = np.random.default_rng(seed)

    # sampling uniformly
    dim = x.shape[1]
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    samples_uniform = rng.uniform(
        low=mins,
        high=maxs,
        size=(n_samples, dim)
    )

    # getting data points from samples
    indices, distances = pairwise_distances_argmin_min(samples_uniform, x)
    samples = x[indices]

    return samples, indices, distances


def LHS_sampling(x, n_samples, seed):
    # sampling space
    dim = x.shape[1]
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    sampler = qmc.LatinHypercube(d=dim, rng=seed)
    samples_lhs = sampler.random(n=n_samples)
    samples_lhs_rescaled = qmc.scale(samples_lhs, mins, maxs)  # bring LHS into PCA box

    # getting datapoints
    indices, distances = pairwise_distances_argmin_min(samples_lhs_rescaled, x)
    samples = x[indices]

    return samples, indices, distances


def gaussian_sample(x, n_samples, seed):
    rng = np.random.default_rng(seed)
    
    # sampling space
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    samples_gaussian = rng.multivariate_normal(mean, cov, size=n_samples)

    # getting datapoints
    indices, distances = pairwise_distances_argmin_min(samples_gaussian, x)
    samples = x[indices]

    return samples, indices, distances


def min_max_sample(x, n_samples, seed):
    rng = np.random.default_rng(seed)

    # sampling space
    n = x.shape[0]
    selected = []
    init_index = rng.integers(n)
    selected.append(init_index)

    # Compute initial distances
    min_distances = np.linalg.norm(x - x[init_index], axis=1)  # distances from first point
    for _ in range(n_samples-1):
        min_distances[selected[-1]] = -np.inf  # exclude latest point
        farthest = np.argmax(min_distances)
        selected.append(farthest)

        # Update minimum distances
        dists = np.linalg.norm(x - x[farthest], axis=1)
        min_distances = np.minimum(min_distances, dists)

    # getting datapoints
    indices = np.array(selected)
    distances = None # no notion of distance, actually picking points
    samples = x[indices]

    return samples, indices, distances


def kmeans_sample(x, n_samples, seed):
    # sampling space
    kmeans = KMeans(n_clusters=n_samples, random_state=seed).fit(x)
    centers = kmeans.cluster_centers_

    # getting datapoints
    indices, distances = pairwise_distances_argmin_min(centers, x)
    samples = x[indices]

    return samples, indices, distances


def gmm_sample(x, n_samples, seed, n_gaussians=100):
    # sampling space
    gmm = GaussianMixture(
        n_components=n_gaussians,
        covariance_type='full',
        random_state=seed
    )
    gmm.fit(x)
    samples_gmm, _ = gmm.sample(n_samples)

    # getting datapoints
    indices, distances = pairwise_distances_argmin_min(samples_gmm, x)
    samples = x[indices]

    return samples, indices, distances


def sample_with_all_methods(x, n_samples, seed=12345, n_gaussians=100):
    """
    samples data with every sampling method

    Args:
        x (np.ndarray): data matrix
        seed (int): random seed for sampling
        n_samples (int): number of points to sample
        n_gaussians (int): number of gaussian distributions to use for the GMM

    Returns:
        (dict) samples, indicies of original array, and distances from the
        sampled point (if applicable) for each of the sampling methods
    """
    samples_uniform, indices_uniform, distances_uniform = uniform_sample(x, n_samples, seed)
    samples_LHS, indices_LHS, distances_LHS = LHS_sampling(x, n_samples, seed)
    samples_gaussian, indices_gaussian, distances_gaussian = gaussian_sample(x, n_samples, seed)
    samples_min_max, indices_min_max, distances_min_max = min_max_sample(x, n_samples, seed)
    samples_kmeans, indices_kmeans, distances_kmeans = kmeans_sample(x, n_samples, seed)
    samples_gmm, indices_gmm, distances_gmm = gmm_sample(x, n_samples, seed, n_gaussians=n_gaussians)

    return {
        "uniform": {
            "samples": samples_uniform,
            "indices": indices_uniform,
            "distances": distances_uniform
        },
        "LHS": {
            "samples": samples_LHS,
            "indices": indices_LHS,
            "distances": distances_LHS
        },
        "gaussian": {
            "samples": samples_gaussian,
            "indices": indices_gaussian,
            "distances": distances_gaussian
        },
        "min_max": {
            "samples": samples_min_max,
            "indices": indices_min_max,
            "distances": distances_min_max
        },
        "kmeans": {
            "samples": samples_kmeans,
            "indices": indices_kmeans,
            "distances": distances_kmeans
        },
        "gmm": {
            "samples": samples_gmm,
            "indices": indices_gmm,
            "distances": distances_gmm
        }
    }
