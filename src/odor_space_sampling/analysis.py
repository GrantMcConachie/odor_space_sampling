"""
Scripts for analyzing various sampling methods
"""

import numpy as np
from tqdm import tqdm
from scipy.stats import ks_2samp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

from .data import OdorData
from .utils import get_rd_labels_full
from .plotting import plot_gmm_sweep


def aic_and_bic_gmm(data: OdorData, n_samples=100, max_n_clusters=200, seed=12345, plot=True):
    """
    Calculating AIC and BIC for the GMM with differents amount of clusters

    Args:
        data (OdorData): bundled dataframe and data matrix
        n_samples (int): number of samples to get
        max_n_clusters (int): number of gmms to fit
        seed (int): random seed
        plot (bool): weather to plot the output or not

    Returns:
        aics (list): Akaike information criterion values for each fitted GMM
        bics (list): bayesian information criterion values for each fitted GMM
        ks_means (list): mean Kolmogorov-Smirnov (K-S) statistic between
                         sampled points and full data distribution for all data
                         features
        ks_meds (list): median Kolmogorov-Smirnov (K-S) statistic between
                        sampled points and full data distribution for all data
                        features
    """
    # Get the cdf of the full data
    rd_desc, _ = get_rd_labels_full(data.df['smiles'])  # all data
    all_param = np.array(rd_desc)

    # Doing a sweep over GMM parameters and plotting the AIC this gone take a while
    aics = []
    ks_means = []
    ks_meds = []
    bics = []
    for i in tqdm(range(max_n_clusters), desc="fitting gmms"):
        # fit gmm
        gmm = GaussianMixture(n_components=i+1, covariance_type='full', random_state=seed)
        gmm.fit(data.x)

        # get AIC criteria
        aic = gmm.aic(data.x)
        bic = gmm.bic(data.x)
        aics.append(aic)
        bics.append(bic)

        # sample gaussian
        samples_gmm, _ = gmm.sample(n_samples)
        indices_gmm, _ = pairwise_distances_argmin_min(samples_gmm, data.x)

        # find ks stat of data
        chosen_smiles = data.df.iloc[indices_gmm]['smiles']
        rd_desc, _ = get_rd_labels_full(chosen_smiles)
        params = np.array(rd_desc)
        ks_stats = [ks_2samp(all_param[:,i], params[:,i]).statistic for i in range(all_param.shape[1])]
        mean_val = np.nanmean(ks_stats)
        med_val = np.nanmedian(ks_stats)
        ks_means.append(mean_val)
        ks_meds.append(med_val)

    if plot:
        plot_gmm_sweep(aics, bics, ks_means, ks_meds)

    return aics, bics, ks_means, ks_meds
