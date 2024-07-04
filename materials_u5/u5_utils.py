# -*- coding: utf-8 -*-
"""
Authors: B. Schäfl, S. Lehner, J. Schimunek, J. Brandstetter, A. Schörgenhumer
Date: 18-04-2023

This file is part of the "Hands-on AI II" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import warnings
warnings.filterwarnings(action=r'ignore', category=UserWarning)

import multiprocessing
import sys
from packaging.version import Version
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import PIL
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import seaborn as sns
import sklearn
from IPython.core.display import HTML

from scipy.stats import spearmanr

from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator 
from rdkit.Chem import rdMolDescriptors



def setup_jupyter() -> HTML:
    """
    Setup Jupyter notebook. Warning: this may affect all Jupyter notebooks running on the same Jupyter server.

    :return: HTML instance comprising the modified Jupyter attributes
    """
    return HTML(r"""
    <style>
        .output_png {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
        .jp-RenderedImage {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
    </style>
    <p>Setting up notebook ... finished.</p>
    """)


# noinspection PyUnresolvedReferences
def check_module_versions() -> None:
    """
    Check Python version as well as versions of recommended (partly required) modules.
    """
    python_check = '(\u2713)' if sys.version_info >= (3, 8) else '(\u2717)'
    numpy_check = '(\u2713)' if Version(np.__version__) >= Version('1.18') else '(\u2717)'
    pandas_check = '(\u2713)' if Version(pd.__version__) >= Version('1.0') else '(\u2717)'
    sklearn_check = '(\u2713)' if Version(sklearn.__version__) >= Version('1.2') else '(\u2717)'
    matplotlib_check = '(\u2713)' if Version(matplotlib.__version__) >= Version('3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if Version(sns.__version__) >= Version('0.10.0') else '(\u2717)'
    pil_check = '(\u2713)' if Version(PIL.__version__) >= Version('6.0.0') else '(\u2717)'
    rdkit_check = '(\u2713)' if Version(rdkit.__version__) >= Version('2020.03.4') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed PIL version: {PIL.__version__} {pil_check}')
    print(f'Installed rdkit version: {rdkit.__version__} {rdkit_check}')


def apply_pca(data: pd.DataFrame, n_components: Optional[int] = None, target_column: Optional[str] = None,
              components: Optional[PCA] = None, return_components: bool = False
              ) -> Union[Tuple[pd.DataFrame, PCA], pd.DataFrame]:
    """
    Apply principal component analysis (PCA) on specified dataset and down-project project data accordingly.

    :param data: dataset to down-project
    :param n_components: amount of (top) principal components involved in down-projection
    :param target_column: if specified, append target column to resulting, down-projected data set
    :param return_components: return principal components in addition of down-projected data set
    :param components: use these principal components instead of freshly computing them
    :return: down-projected data set and optionally principal components
    """
    assert type(data) == pd.DataFrame
    assert ((n_components is None) and (components is not None)) or (type(n_components) == int) and (n_components >= 1)
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    assert (components is None) or (type(components) == PCA)
    assert type(return_components) == bool
    
    if target_column is not None:
        target_data = data[target_column]
        data = data.drop(columns=target_column)
    
    if components is None:
        components = PCA(n_components=n_components).fit(data)
    projected_data = pd.DataFrame(components.transform(data), index=data.index)
    if target_column is not None:
        projected_data[target_column] = target_data
    
    return (projected_data, components) if return_components else projected_data


def apply_tsne(n_components: int, data: pd.DataFrame, target_column: Optional[str] = None,
               perplexity: float = 10.0) -> pd.DataFrame:
    """
    Apply t-distributed stochastic neighbor embedding (t-SNE) on specified dataset and down-project data accordingly.

    :param n_components: dimensionality of the embedding space
    :param data: dataset to down-project
    :param target_column: if specified, append target column to resulting, down-projected dataset
    :param perplexity: this term is closely related to the number of nearest neighbors to consider
    :return: down-projected dataset
    """
    assert (type(n_components) == int) and (n_components >= 1)
    assert type(data) == pd.DataFrame
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    assert (type(perplexity) == float) or (type(perplexity) == int)
    if target_column is not None:
        projected_data = pd.DataFrame(TSNE(n_components=n_components, perplexity=float(perplexity), learning_rate=200,
                                           init="random").fit_transform(data.drop(columns=target_column)), index=data.index)
        projected_data[target_column] = data[target_column]
    else:
        projected_data = pd.DataFrame(TSNE(n_components=n_components, perplexity=float(perplexity), learning_rate=200,
                                           init="random").fit_transform(data), index=data.index)
    return projected_data


def apply_k_means(k: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply k-means clustering algorithm on the specified data.

    :param k: amount of clusters
    :param data: data used for clustering
    :return: predicted cluster per dataset entry
    """
    assert (type(k) == int) and (k >= 1)
    assert type(data) == pd.DataFrame
    return KMeans(n_clusters=k, n_init="auto").fit_predict(data)


def apply_affinity_propagation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply affinity propagation clustering algorithm on the specified data.

    :param data: data used for clustering
    :return: predicted cluster per dataset entry
    """
    assert type(data) == pd.DataFrame
    return AffinityPropagation(affinity='euclidean', random_state=0).fit_predict(data)

    
def _compute_ecfps_ecfp_worker(smiles: str, radius: int) -> Dict[int, int]:
    """
    Compute ECFP of a SMILES representation.
    
    :param smiles: SMILES representation for which to compute the ECFP
    :param radius: radius to be used for each atom to compute the ECFP
    :return: ECFP of the specified SMILES representation
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Error parsing SMILES "{smiles}".')
    return AllChem.GetMorganFingerprint(molecule, radius).GetNonzeroElements()


def _compute_ecfps_fold_worker(ecfp: Dict[int, int], lookup: Dict[int, int], fold: int) -> np.ndarray:
    """
    Expand and fold an ECFP using a specified lookup table.
    
    :param ecfp: ECFP to expand and fold
    :param lookup: lookup table to be used to fold ECFP
    :param fold: minimum length of expanded and folded ECFP
    :return: expanded and folded ECFP
    """
    resulting_fold = max(fold, max(ecfp.values()))
    result = np.zeros(shape=(1, resulting_fold), dtype=bool)
    for key, value in ecfp.items():
        result[0, lookup[key]] = True
    return result
    

def compute_ecfps(smiles: Sequence[str], radius: int = 3, fold: int = 1024, num_jobs: int = 0) -> pd.DataFrame:
    """
    Compute ECFPs of specified SMILES representations.
    
    :param smiles: SMILES representations for which to compute the ECFPs
    :param radius: radius to be used for each atom to compute the ECFPs
    :param fold: minimum length of expanded and folded ECFPs
    :param num_jobs: amount of parallel processes to be used for computing ECFPs (<= 0: all available cores)
    :return: computed ECFPs, expanded and folded
    """
    assert (type(smiles) in (list, tuple)) and (len(smiles) > 0)
    assert all((type(_) == str) for _ in smiles)
    assert (type(radius) == int) and (radius >= 0)
    assert (type(fold) == int) and (fold > 0)
    assert type(num_jobs) == int
    
    # Compute ECFPs of specified SMILES representations in a multi-processing manner.
    with multiprocessing.Pool(multiprocessing.cpu_count() if num_jobs <= 0 else num_jobs) as worker_pool:
        ecfps = worker_pool.map(partial(_compute_ecfps_ecfp_worker, radius=radius), smiles)
    
        # Parse computed ECFPs.
        fold_mapping = {key: key % fold for key in set.union(*(set(ecfp.keys()) for ecfp in ecfps))}
        ecfps = worker_pool.map(partial(_compute_ecfps_fold_worker, lookup=fold_mapping, fold=fold), ecfps)
    
    # Combine computed ECFPs.
    ecfps = np.concatenate(ecfps)
    return pd.DataFrame(ecfps)


def plot_points_2d(data: pd.DataFrame, target_column: Optional[str] = None, targets: Sequence = None,
                   legend: bool = True, multi_color_palette: str = "husl",  **kwargs) -> None:
    """
    Visualize data points in a two-dimensional plot, optionally color-coding according to ``target_column``.

    :param data: dataset to visualize
    :param target_column: optional target column to be used for color-coding
    :param targets: sequence of target labels if not contained in ``data`` (via ``target_column``)
    :param legend: flag for displaying a legend
    :param multi_color_palette: Seaborn color palette to use when > 10 targets are plotted
    :param kwargs: optional keyword arguments passed to ``plt.subplots``
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] in [2, 3])
    assert not (target_column is not None and targets is not None), "can only specify either 'target_column' or 'targets"
    assert (target_column is None) or ((data.shape[1] == 3) and (data.columns[2] == target_column))
    assert targets is None or len(targets) == len(data)
    fig, ax = plt.subplots(**kwargs)
    color_targets = None
    hue_order = None
    if target_column is not None:
        color_targets = data[target_column]
    elif targets is not None:
        color_targets = targets
    color_palette = None
    if color_targets is not None:
        n_colors = len(set(color_targets))
        palette = multi_color_palette if multi_color_palette else "muted"
        color_palette = sns.color_palette(palette=palette, n_colors=n_colors)
        hue_order = sorted(list(set(color_targets)))
    legend = "auto" if legend else False
    sns.scatterplot(x=data[0], y=data[1], hue=color_targets, ax=ax, palette=color_palette, legend=legend, hue_order=hue_order)
    plt.tight_layout()
    plt.show()


def apply_sphere_exclusion_clustering(data, th=0.65):
    """
    Apply Sphere exclusion clustering algorithm to binary feature vectors.

    :param data: data used for clustering
    :return: predicted cluster per dataset entry
    """

    fps = [DataStructs.cDataStructs.CreateFromBitString("".join(fp_vec.astype(str))) for fp_vec in data.values]
    lp = rdSimDivPickers.LeaderPicker()
    picks = lp.LazyBitVectorPick(fps,len(fps),th)
    pickfps = [fps[x] for x in picks]
    cluster_ids = []
    for fp in fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, pickfps)
        cluster_ids.append(np.argmax(sims))
    return cluster_ids

def bin_contineous_label_values(x: list, num_bins: int =5):
    """
    Create binned lables from list of contineous label values.

    :param x: list of contineous values
    :param num_bins: number of bins
    :return: list of binned labels
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(x, percentiles)
    bin_edges[-1] += 0.1
    inds = np.digitize(x, bin_edges) -1 
    labels = [f'[{bin_edges[i]:.1f},{bin_edges[i+1]:.1f}[' for i in inds]        
    return labels

def apply_cluster_split(df: pd.DataFrame, cluster_column: str = 'Cluster', test_size: float = 0.25):
    """
    Create train test split based on clusters.

    :param df: pd.DataFrame containing dataset
    :param cluster_column: column containing cluster indices
    :param test_size: approximate size of test set
    :return df_train, df_test: train and test set data
    """
    # Calculate the size of each cluster
    cluster_size = df[cluster_column].value_counts()
    
    
    train_cluster_indices = []
    train_size = 0
    n = df.shape[0]

    # Iterate over each cluster, add clusters into training set until training set limit size reached
    for cluster_id, size in cluster_size.items():
        train_cluster_indices.append(cluster_id)
        train_size += size/n
        if np.round(train_size,2) >= np.round(1-test_size,2):
            break

    df_train = df.loc[df[cluster_column].isin(train_cluster_indices)]
    df_test = df.loc[~df[cluster_column].isin(train_cluster_indices)]

    return df_train, df_test

def train_and_evaluate_sklearn_regressor(
        model, 
        X_train: np.array, 
        y_train: np.array, 
        X_test: np.array, 
        y_test: np.array, 
        low_variance_feature_th: float | None = None,
        metric: callable = mean_squared_error,
        plot: bool = True,
        verbose: bool = True):

    if low_variance_feature_th is not None:
        # Drop features with zero variance
        selector = VarianceThreshold(low_variance_feature_th)
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test)
        if verbose:
            print('Number of features:', X_train.shape[1])

    # Train model
    model.fit(X_train, y_train)

    # Make train and test set predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    if metric != None:
        train_metric = metric(train_preds, y_train)
        test_metric = metric(test_preds, y_test)
        metric_name = metric.__name__
        if verbose:
            print(f'Train {metric_name}: {train_metric:.3f}')
            print(f'Test {metric_name}: {test_metric:.3f}')

    if plot:
        _, ax = plt.subplots(1,2, figsize=(9, 4), sharex=True, sharey=True)
        
        axmin = min(min(y_train), min(y_test), min(train_preds), min(test_preds))
        axmax = max(max(y_train), max(y_test), max(train_preds), max(test_preds))

        sns.regplot(x=y_train, y=train_preds, ax=ax[0], label='Train')
        sns.regplot(x=y_test, y=test_preds, ax=ax[1], label='Test', color='orange')

        for a in ax:
            a.plot([axmin, axmax], [axmin, axmax], 'k--')
            a.legend()

        ax[0].set_xlabel('True value')
        ax[0].set_ylabel('Predicted value')

    return train_preds, test_preds, train_metric, test_metric


def plot_feature_fractions(df):

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

    # Plot the line plot of the mean values on the left subplot
    mean_values = df.mean(axis=0)
    ax1.plot(mean_values, marker='o', ms=2, mec='k', linewidth=0)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Fraction')
    ax1.set_ylim(-0.05,1.05)

    # Plot the histogram of the y values on the right subplot
    ax2.hist(mean_values, bins=np.arange(0.0001,1.1,0.01), orientation='horizontal', edgecolor='black')
    ax2.hist(mean_values, bins=np.arange(-0.001,0.002,0.001), orientation='horizontal', edgecolor='red')
    ax2.set_xlabel('Frequency')
    ax2.set_xscale('log')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def load_molecules_from_csv(file: str, smiles_col: str = 'SMILES', prop_cols: list[str] | str = None):

    # Load molecules
    df = pd.read_csv(file)
    df['mol'] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))

    # Add chemical formula
    df['Formula'] = df.mol.apply(lambda x : rdMolDescriptors.CalcMolFormula(x))
    df = df.sort_values(by='Formula')

    # Property columns
    prop_cols = [smiles_col, 'Formula'] + prop_cols if prop_cols != None else df.columns
    prop_cols = [c for c in prop_cols if c != 'mol']

    # Add properties to molecules
    for prop in prop_cols:
        props = df[prop].tolist()
        # mols = [mol.SetProp(prop, str(x)) for mol, x in zip(mols, props)]
        for mol, x in zip(df.mol, props):
            if mol is not None:
                mol.SetProp(prop, str(x))
            else:
                print(mol)

    return df.mol.tolist()


def spearmanr_score(x,y):
    """Spearman rank correlation coefficient between x and y"""
    return spearmanr(x,y)[0]