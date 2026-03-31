"""
utility functions for processing the data
"""

import numpy as np
from tqdm import tqdm
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def remove_nans(mat):
    """
    removes nan values from the data matrix

    Args:
        mat (np.ndarray): (num smiles, desctitor) matrix
    
    Returns:
        cleaned_mat (np.ndarray): cleaned, no nan matrix
    """
    nan_cols = np.unique(np.where(np.isnan(mat))[1])
    cleaned_mat = np.delete(mat, nan_cols, axis=1)
    return cleaned_mat


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

    # taking out zero variance columns and nans
    x = remove_nans(x)
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


def make_rdkit_descriptors(df):
    """
    Converts loaded dataframe into rdkit descriptors

    Args:
        df (str): dataframe of csv
    
    Returns:
        data_matrix (np.ndarray): (num smiles, descriptor) matrix
    """
    smiles_list = list(df['smiles'])

    # embed odors into rdkit descriptor vectors
    data_matrix = []
    for smi in tqdm(smiles_list, desc="generating descriptors"):
        mol = Chem.MolFromSmiles(smi)
        descriptor_dict = Descriptors.CalcMolDescriptors(mol)
        desc_vector = list(descriptor_dict.values())
        data_matrix.append(desc_vector)

    data_matrix = np.array(data_matrix)

    return data_matrix


def get_rd_fun_group_labels(smiles_list):
    """
    Gets the rdkit descriptors for labeling molecules by their functional
    groups

    Args:
        smiles_list (list) - List of all the smiles

    Returns:
        rd_desc (list) - List of all the rd descriptors
    """
    rd_desc = []
    list_rd_desc = [x[0] for x in Descriptors._descList if "fr" in x[0]]
    calc_rd = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList if "fr" in x[0]]
    )
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        rd_result = np.array(calc_rd.CalcDescriptors(mol))
        rd_desc.append(rd_result)

    return rd_desc, list_rd_desc


def get_ks_stats(df1, df2):
    """
    Gets the median and mean of all the KS statistics each rdkit feature of 
    two lists of smiles.

    Args:
        df1, df2 (pandas.DataFrame): dataframes with a column of smiles
    
    Returns:
        ks_values (list): The ks-statistics for all the rdkit features
        ks_mean (float): mean value of all ks statistics
        ks_median (float): median value of all the ks statistics
    """
    # extract smiles and rd descriptors
    df1_smiles = list(df1['smiles'])
    df2_smiles = list(df2['smiles'])
    df1_rd_desc, _ = get_rd_labels_full(df1_smiles)
    df2_rd_desc, _ = get_rd_labels_full(df2_smiles)
    df1_rd_desc = np.array(df1_rd_desc)
    df2_rd_desc = np.array(df2_rd_desc)

    # calculate all the ks statistics
    ks_values = []
    for i in range(df2_rd_desc.shape[1]):
        ks_values.append(ks_2samp(df1_rd_desc[:, i], df2_rd_desc[:, i]))
    
    # get mean and median
    ks_mean = np.nanmean(ks_values)
    ks_median = np.nanmedian(ks_values)

    return ks_values, ks_mean, ks_median


def get_num_fn_groups(df, sample_methods, save_path=None):
    """
    Gets the number of all functional groups in a dataframe.

    Args:
        df (pd.DataFrame): dataframe with 'smiles' column
        sample_methods (list[tuple]): list of (indices, label) pairs
        save_path (str, optional): if provided, saves missing functional groups
            per method to a txt file at this path

    Returns:
        labels (list): 
    """
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
    
    return labels, num_fr, missing_groups, desc_names
