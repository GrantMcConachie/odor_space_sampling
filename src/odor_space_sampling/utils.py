"""
utility functions for processing the data
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def remove_zero_var_descriptors(x):
    """
    removes columns with zero variance
    
    Args:
        x (np.ndarray): data matrix
    
    Returns:
        (np.ndarray) datamatrix with no zero variance columns
    """
    mask = x.std(axis=0) > 1e-6
    return x[:, mask]


def zscore_features(x):
    """
    zscore features

    Args:
        x (np.ndarray): data matrix
    
    Returns:
        (np.ndarray) datamatrix with with zscored columns
    """
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def reduce_data(x):
    """
    takes out all zero variance features, z-scores each of them, and reduces 
    data matrix with PCA to 99% explained variance

    Args:
        x (np.ndarray): data matrix with shape (num smiles, descriptors)
    
    Returns:
        x_reduced (np.ndarray): dimensionailty reduced data
    """
    print(f'initial data shape: {x.shape}')

    # taking out zero variance columns
    x = remove_zero_var_descriptors(x)
    print("dimension after removing constant features:", x.shape)

    # Standardize features (zero mean, unit variance)
    x = zscore_features(x)

    # run pca and reduce to 99% variance
    pca = PCA()
    pca.fit(x)

    # find 99% variance and reduce
    dim = np.where(np.cumsum(pca.explained_variance_ratio_)>0.99)[0][0] + 1
    print(f'dimensionality of 99% explained variance: {dim}')
    pca = PCA(n_components=dim)
    x_reduced = pca.fit_transform(x)
    print(f'reduced space shape {x_reduced.shape}')

    return x_reduced


def get_rd_labels_full(smiles_list):
    """
    Gets the rdkit descriptors for labeling molecules by their functional
    groups

    Args:
        smiles_list (list): List of all the smiles

    Returns:
        rd_desc (list): List of values of all rdkit descriptors
        list_rd_desc (list): List of all the rd descriptors
    """
    rd_desc = []
    list_rd_desc = [x[0] for x in Descriptors._descList]
    calc_rd = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList]
    )
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        rd_result = np.array(calc_rd.CalcDescriptors(mol))
        rd_desc.append(rd_result)

    return rd_desc, list_rd_desc
