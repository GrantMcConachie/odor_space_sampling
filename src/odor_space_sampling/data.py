"""
Loading and converting data into proper format for sampling
"""

import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors


def load_csv(filepath, sep=','):
    """
    loads csv

    Args:
        filepath (str): path to csv
        sep (str): delimiter used for the data (default comma)
    
    Returns:
        pandas.dataframe of the data
    """
    df = pd.read_csv(filepath, sep=sep)
    assert 'smiles' in df.columns, "DataFrame must contain a 'smiles' column"
    return df


def make_rdkit_descriptors(filepath, sep=','):
    """
    Converts loaded dataframe into rdkit descriptors

    Args:
        filepath (str): path to csv
        sep (str): delimiter used for the data (default comma)
    
    Returns:
        data_matrix (np.ndarray): (num smiles, descriptor) matrix
    """
    df = load_csv(filepath=filepath, sep=sep)
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


def add_cid_to_data(filepath, sep=',', save=False):
    """
    Queries pubchem to add the cid to the dataframe that you're interested in

    Args:
        filepath (str): filepath to the csv of interest
        sep (str): delimiter for the csv (default comma)
        save (bool): overwrites the df with a df that has CID and IUPAC name in
                     it
    
    Returns:
        df (pandas.dataframe): dataframe with CID and IUPAC columns added
    """
    # load csv
    df = load_csv(filepath=filepath, sep=sep)

    # ignore if CID and IUPAC name alreadt in df
    if 'cid' in df.columns and 'iupac' in df.columns:
        print('cid and IUPAC names already in csv, not generating CIDs')
        return df

    # loop through df and get CIDs
    cids = []
    iupac_names = []
    for _, row in tqdm(df.iterrows(), desc="generating CIDs"):
        smiles = row['smiles']
        api_safe_smiles = smiles.encode('unicode_escape').decode()

        # getting CID
        try:
            response = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{api_safe_smiles}/json')
            response = response.json()
            cid = response['PC_Compounds'][0]['id']['id']['cid']
        except:
            cid = 'unknown'

        # getting IUPAC NAME
        try:
            for j in response['PC_Compounds'][0]['props']:
                iupac_name = 'unknown'
                if j['urn']['label'] == "IUPAC Name":
                    iupac_name = j['value']['sval']
                    break

        except:
            iupac_name = 'unknown'

        cids.append((smiles, cid))
        iupac_names.append((smiles, iupac_name))

    # add in cid column
    df['cid'] = [i[1] for i in cids]
    df['IUPAC'] = [i[1] for i in iupac_names]

    if save:
        df.to_csv(filepath, index=False)

    return df
