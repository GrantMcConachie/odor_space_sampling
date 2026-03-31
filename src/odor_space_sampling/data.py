"""
Loading and converting data into proper format for sampling
"""

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass

from .utils import reduce_data, make_rdkit_descriptors


@dataclass
class OdorData:
    """
    Bundles the raw dataframe and its row-aligned data matrix together.

    Attributes:
        df (pd.DataFrame): original dataframe (smiles, label, cid, IUPAC, ...)
        x (np.ndarray): processed data matrix, row i corresponds to df.iloc[i]
    """
    df: pd.DataFrame
    x: np.ndarray


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


def load_and_prepare(filepath, sep=',') -> OdorData:
    """
    Loads a csv and builds the fully processed OdorData object (rdkit
    descriptors -> nan removal -> PCA reduction).

    Args:
        filepath (str): path to csv with a 'smiles' column
        sep (str): delimiter (default comma)

    Returns:
        OdorData with df and row-aligned reduced data matrix x
    """
    df = load_csv(filepath, sep=sep)
    x = make_rdkit_descriptors(df)
    x = reduce_data(x)
    return OdorData(df=df, x=x)


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


def create_indices(fp, reference_fp, fp_sep=',', reference_fp_sep=','):
    """
    Returns integer indices into a reference dataframe for each row in a subset
    dataframe, matched on the 'smiles' column.

    Useful for passing an externally-sourced sample CSV into plotting functions
    (plot_sampling_projections, plot_all_sampling_methods_coverage,
    plot_all_sampling_methods_fun_groups, plot_all_sampling_methods_data_dist,
    plot_ks_dist) that expect indices into the full reference OdorData.

    Args:
        fp (str): path to the subset (sampling) CSV
        reference_fp (str): path to the reference (full) CSV
        fp_sep (str): delimiter for the subset CSV (default comma)
        reference_fp_sep (str): delimiter for the reference CSV (default comma)

    Returns:
        np.ndarray: integer indices into the reference dataframe, one per matched row

    Raises:
        ValueError: if a SMILES string in the subset is not found in the reference

    Examples:
        indices = create_indices('my_sample.csv', 'reference.csv')
        plot_sampling_projections(reference_data, (indices, "My Sample"))

        plot_all_sampling_methods_coverage(
            reference_data, results,
            extra_methods=[(indices, "My Sample")]
        )
    """
    df = load_csv(fp, fp_sep)
    ref_df = load_csv(reference_fp, reference_fp_sep)

    smiles_to_idx = {s: i for i, s in enumerate(ref_df['smiles'])}
    indices = []
    for s in df['smiles']:
        if s not in smiles_to_idx:
            raise ValueError(f"SMILES '{s}' from subset not found in reference dataframe")
        indices.append(smiles_to_idx[s])

    return np.array(indices)
