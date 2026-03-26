"""
functions for sampling odor space
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

from .data import OdorData


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


def sample_with_all_methods(data, n_samples, seed=12345, n_gaussians=100):
    """
    samples data with every sampling method

    Args:
        x (OdorData): bundled dataframe and data matrix
        seed (int): random seed for sampling
        n_samples (int): number of points to sample
        n_gaussians (int): number of gaussian distributions to use for the GMM

    Returns:
        (dict) samples, indicies of original array, and distances from the
        sampled point (if applicable) for each of the sampling methods
    """
    x = data.x
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


def get_n_closest_points_gmm(
        data: OdorData,
        n_closest_points,
        seed=12345,
        n_clusters=100,
        n_samples=100,
        save_path=None
):
    """
    Fits a gmm and calculates the n closest points to each sampled point

    Args:
        data (OdorData): bundled dataframe and data matrix
        n_samples (int): number of points to sample from the GMM
        seed (int): random seed
        n_clusters (int): number of gaussian components in the GMM
        n_closest_points (int): number of nearest real odors to return per sample
        save_path (str): if provided, saves the output df to this path

    Returns:
        (pd.DataFrame) with columns: sample, smiles, label, cid, IUPAC
    """
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type='full', random_state=seed
    )
    gmm.fit(data.x)
    samples_gmm, _ = gmm.sample(n_samples)
    pair_dist = pairwise_distances(samples_gmm, data.x)
    closest_points = np.argsort(pair_dist, axis=1)

    rows = []
    for i, locs in enumerate(closest_points):
        for j in range(n_closest_points):
            row = data.df.iloc[locs[j]]
            rows.append({
                'sample': i,
                'smiles': row['smiles'],
                'label': row['label'],
                'cid': row['cid'],
                'IUPAC': row['IUPAC'],
            })

    result_df = pd.DataFrame(rows)

    if save_path is not None:
        result_df.to_csv(save_path, index=False)

    return result_df


def gmm_resample_varying_seeds(
        data: OdorData,
        seeds,
        n_clusters=100,
        n_samples=100,
        save_path=None
):
    """
    Function that creates multiple GMMs given a set of random seeds and samples
    odors with each seed.

    Args:
        data (OdorData): bundled dataframe and data matrix
        seeds (list[int]): seeds to initialize the gmm
        n_samples (int): number of points to sample from the GMM
        n_clusters (int): number of gaussian components in the GMM
        save_path (str): if provided, saves the output df to this path
    """
    dfs = []
    if '.csv' in save_path:
        save_path = save_path.replace('.csv', '')

    for seed in tqdm(seeds, desc="iterating through seeds"):
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=seed)
        gmm.fit(data.x)

        # Sample from the GMM in PCA space
        samples_gmm, _ = gmm.sample(n_samples)
        indices_gmm, _ = pairwise_distances_argmin_min(samples_gmm, data.x)
        
        df_to_save = data.df.iloc[indices_gmm]
        dfs.append(df_to_save)

        if save_path is not None:
            df_to_save.to_csv(f'{save_path}_{seed}.csv', index=False)
