# -*-coding:utf-8 -*-
'''
@Time    :   2023/09/14 21:51
@Author  :   Jialong Jiang, Yingying Gong
'''

from scipy.linalg import orth
from collections import deque
import numba
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF, PCA
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import os
from sklearn.cluster import KMeans
import scipy.io as sio
from tqdm import tqdm
from typing import Tuple, List, Callable, Any, Dict
import warnings
from scipy.sparse import issparse
from joblib import Parallel, delayed, parallel_backend


def category_balance_number(
        total_sample_size: int,
        cluster_count: List[int],
        method: str,
        maximum_sample_rate: float) -> np.array:
    """
    Calculate the sampling number for each category based on the specified method.

    Parameters
    ----------
    total_sample_size : int
        The total number of samples.
    cluster_count : List[int]
        A list containing the count of elements in each cluster.
    method : str
        The method used for balancing the categories. Options are 'equal', 'proportional', or 'squareroot'.
    maximum_sample_rate : float
        The maximum sample rate allowed.

    Returns
    -------
    np.array
        An array containing the number of samples for each category computed using the given method.
    """

    # Validate the method parameter
    if method not in ['equal', 'proportional', 'squareroot']:
        raise ValueError('method must be one of equal, proportional, squareroot')

    # Calculate the sampling number based on the selected method.
    if method == 'squareroot':
        # For the 'squareroot' method, compute values based on the square root of the cluster counts.
        esti_size = (np.sqrt(cluster_count) / np.sum(np.sqrt(cluster_count)) * total_sample_size).astype(int)
        weight_fun = np.min([esti_size, maximum_sample_rate * np.array(cluster_count)], axis=0)
    elif method == 'equal':
        # For the 'equal' method, divide the total sample size equally among clusters.
        esti_size = total_sample_size / len(cluster_count)
        weight_fun = np.min([esti_size * np.ones(len(cluster_count)), maximum_sample_rate * np.array(cluster_count)], axis=0)
    else:
        # For the 'proportional' method, use the cluster counts directly as weights.
        weight_fun = cluster_count

    # Compute the final sampling number for each category.
    sampling_number = (weight_fun / np.sum(weight_fun) * total_sample_size).astype(int)
    return sampling_number


from sklearn.cluster import AgglomerativeClustering
import leidenalg as la
import igraph as ig
import networkx as nx

def summary_components(all_components: np.array,
                       num_spin: int,
                       summary_method: str = 'kmeans',
                       figure_folder: str=None) -> List[np.array]:
    """
    Summarize components using a clustering algorithm.

    Parameters
    ----------
    all_components : np.array
        A 2D array where each row represents a sample and each column represents a feature.
    num_spin : int
        The number of clusters.
    summary_method : str, optional
        The method used for summarizing the components. Options are 'kmeans' or 'leiden'. Default is 'kmeans'.
    figure_folder : str, optional
        Folder in which to save the figure.

    Returns
    -------
    List[np.array]
        A list of numpy arrays, each containing the indices of the genes that belong to a specific cluster.
    """
    num_gene = all_components.shape[1]

    if summary_method == 'kmeans':
        # Fit KMeans clustering to the components and to their transposed version.
        kmeans = KMeans(n_clusters=num_spin, random_state=0, n_init=50).fit(all_components)

        # take the genes with larger weights for display only
        gene_weight = np.sum(all_components ** 2, axis=0) ** 0.5
        gene_sele_filt = gene_weight > np.mean(gene_weight)
        all_components_sub = all_components[:, gene_sele_filt]
        kmeans_gene = KMeans(n_clusters=min(2 * num_spin, all_components_sub.shape[1]), random_state=0, n_init=10).fit(all_components_sub.T)

        # Initialize an array to store the average component for each cluster.
        components_kmeans = np.zeros((num_spin, num_gene))

        for ii in range(num_spin):
            # Calculate the average component for the current cluster.
            components_kmeans[ii] = np.mean(all_components[kmeans.labels_ == ii], axis=0)

        # Normalize the computed components.
        components_kmeans = normalize(components_kmeans, axis=1, norm='l2')

        gene_groups_ind = []
        for ii in range(num_spin):
            # Determine gene cluster assignments based on the maximum component values.
            gene_groups_ind.append(np.argmax(components_kmeans, axis=0) == ii)

        

        sc.set_figure_params(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        gene_order = np.argsort(kmeans_gene.labels_)
        comp_order = np.argsort(kmeans.labels_)
        plt.imshow(all_components_sub[comp_order, :][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(all_components) / 10, interpolation='none')
        plt.title('All components')

        plt.subplot(1, 2, 2)
        plt.imshow(components_kmeans[:, gene_sele_filt][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(components_kmeans) / 10, interpolation='none')
        plt.title('Kmeans components')

        if figure_folder is not None:
            plt.savefig(figure_folder + 'onmf_decomposition_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    elif summary_method == 'leiden':

        consensus = np.einsum('ij,ik->jk', all_components, all_components) / all_components.shape[0]
        consensus_filt = np.where(np.max(consensus, axis=0) > np.percentile(consensus.flatten(), 99))[0]
        consensus_sub = consensus[:, consensus_filt][consensus_filt]

        # Perform Agglomerative Clustering on consensus matrix
        clusterer = AgglomerativeClustering(n_clusters=num_spin, metric='precomputed', linkage='complete')
        cluster_labels = clusterer.fit_predict(np.sqrt(- np.log(consensus_sub)))

        # Convert to graph and apply Leiden algorithm
        G = nx.from_numpy_array(consensus)
        G = ig.Graph.from_networkx(G)
        np.random.seed(0)

        optimiser = la.Optimiser()
        membership = np.ones(consensus.shape[0]) * (1 + max(cluster_labels))
        membership[consensus_filt] = cluster_labels
        membership_fixed = np.zeros(consensus.shape[0])
        membership_fixed[consensus_filt] = 1

        membership = membership.astype(int)
        membership_fixed = membership_fixed.astype(int)

        partition = la.RBConfigurationVertexPartition(G, initial_membership=membership, weights='weight')
        diff = optimiser.optimise_partition(partition, n_iterations=- 1, is_membership_fixed=list(membership_fixed))

        gene_groups_ind = []
        cluster_label_full = np.nan * np.ones(consensus.shape[0])

        for ii, part in enumerate(partition):
            gene_groups_ind.append(part)
            cluster_label_full[part] = ii

        gene_order = np.argsort(cluster_label_full)
        gene_list_size = np.array([np.sum(cluster_label_full == ii) for ii in range(num_spin)])

        plt.figure(figsize=[8, 8])
        plt.imshow(consensus[gene_order][:, gene_order], cmap='Blues', vmin=0, vmax=np.percentile(consensus.flatten(), 98))
        for ii in range(1, num_spin):
            plt.axvline(np.sum(gene_list_size[:ii]) - 0.5, color='k')
            plt.axhline(np.sum(gene_list_size[:ii]) - 0.5, color='k')
        plt.grid()

        if figure_folder is not None:
            plt.savefig(figure_folder + 'onmf_decomposition_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

    return gene_groups_ind


def onmf(X: np.array, rank: int, max_iter: int = 500) -> Tuple[np.array, np.array]:
    """
    Perform Orthogonal Non-Negative Matrix Factorization (ONMF) for a given rank.

    Parameters
    ----------
    X : np.array
        Input data matrix.
    rank : int
        Desired rank for factorization.
    max_iter : int, optional
        Maximum number of iterations. Default is 500.

    Returns
    -------
    Tuple[np.array, np.array]
        The factorized matrix S.T.
        The matrix A.
    """

    m, n = X.shape

    # Initialize matrices A and S
    A = np.random.rand(m, rank) 
    S = np.random.rand(rank, n)
    S = np.abs(orth(S.T).T)

    pbar = tqdm(total=max_iter, desc="Iteration Progress")

    for itr in range(max_iter):
        # Update A and S using multiplicative update rules
        coef_A = X.dot(S.T) / A.dot(S.dot(S.T))
        A = np.nan_to_num(A * coef_A) #, posinf=1e5)

        AtX = A.T.dot(X)
        coef_S = AtX / S.dot(AtX.T).dot(S)
        S = np.nan_to_num(S * coef_S) #, posinf=1e5)

        pbar.update(1)

        # Calculate the reconstruction error every 10 iterations and update the
        # progress bar
        if itr % 10 == 0:
            error = np.linalg.norm(X - np.dot(A, S), 'fro')
            self_product = S.dot(S.T)
            orthogonal_error = np.linalg.norm(self_product - np.diag(np.diag(self_product)), 'fro') / np.linalg.norm(S, 'fro')
            pbar.set_postfix({"Reconstruction error": f"{error:.2f}", "Orthogonal error": f"{orthogonal_error:.4f}"})

    pbar.close()

    # Normalize the components
    norm_fac = np.sqrt(np.diag(S.dot(S.T)))
    S /= norm_fac.reshape(-1, 1)
    A *= norm_fac.reshape(1, -1)
    A *= np.sum(X * A.dot(S)) / np.sum((A.dot(S)) ** 2)

    return S.T, A


def compute_onmf(seed: int, 
                 num_spin: int, 
                 gene_matrix: np.array,
                 max_iter: int = 500) -> NMF:
    """
    Compute the ONMF model for the given gene matrix.

    Parameters
    ----------
    seed : int
        Seed for random number generation.
    num_spin : int
        The number of desired components (clusters/spins).
    gene_matrix : np.array
        Matrix representing gene expression data. Typically, the matrix should be normalized by library size and log1-transformed. Normalizing each gene by its standard deviation is also recommended.
    max_iter : int, optional
        Maximum number of iterations. Default is 500.

    Returns
    -------
    NMF
        The NMF model with computed components.
    """
    # Generate a random seed
    np.random.seed(seed)
    # Factorized Matrices
    H, W = onmf(gene_matrix, num_spin, max_iter)

    # Initialize the NMF model
    nmf_model = NMF(n_components=num_spin, random_state=seed)

    # Set the components and number of components
    nmf_model.components_ = np.array(H).T
    nmf_model.n_components_ = num_spin

    return nmf_model


def onmf_discretize(onmf_rep_ori: np.array, 
                    num_init: int = 10,
                    fig_folder: str = None) -> np.array:
    """
    Discretize the ONMF representation using KMeans clustering and visualize the sorted representations.

    Parameters
    ----------
    onmf_rep_ori : np.array
        Original ONMF representation.
    fig_folder : str, optional
        Folder in which to save the figure.

    Returns
    -------
    np.array
        Discretized ONMF representation.
    """
    num_spin = onmf_rep_ori.shape[1]
    sc.set_figure_params(figsize=[2, 2])
    _, grid = sc.pl._tools._panel_grid(
        0.3, 0.3, ncols=7, num_panels=min(
            21, num_spin))

    onmf_rep_tri = np.zeros(onmf_rep_ori.shape)
    print('Processing gene or program expression levels...')
    for ii in tqdm(range(num_spin)):
        # Perform KMeans clustering for each spin/component
        km_fit = KMeans(n_clusters=3, n_init=num_init).fit(onmf_rep_ori[:, ii].reshape(-1, 1))

        onmf_rep_tri[:, ii] = (km_fit.labels_ == np.argsort(km_fit.cluster_centers_.reshape(-1))[0]) * \
            (-1) + (km_fit.labels_ == np.argsort(km_fit.cluster_centers_.reshape(-1))[2]) * 1

        # Visualize the sorted representations
        if ii < 21:
            ax = plt.subplot(grid[ii])
            plt.plot(np.sort(onmf_rep_ori[:, ii]))
            plt.plot(np.sort(km_fit.cluster_centers_[km_fit.labels_].reshape(-1)))

    # Save the visual representation
    if fig_folder is not None:
        print('Saving the example state partition figure to ' + fig_folder + 'onmf_discretize.png')
        plt.savefig(f'{fig_folder}/onmf_discretize.png', bbox_inches='tight')
        plt.close()

    return onmf_rep_tri


def compute_onmf_decomposition(gene_matrix_norm: np.ndarray,
                               num_onmf_components: int,
                               seed: int = 0,
                               params: dict = None,
                               save_path: str = None) -> None:
    """
    Perform multiple oNMF decompositions on subsampled data to enhance stability.

    Parameters
    ----------
    gene_matrix_norm : np.ndarray
        Normalized gene expression matrix.
    num_onmf_components : int
        Number of oNMF components.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    params : dict, optional
        Additional parameters for oNMF. Default is None.
    save_path : str, optional
        Path to save the oNMF decomposition results. Default is None.
    """
    np.random.seed(seed)
    print(f'Computing oNMF decomposition of seed {seed}')

    # Check if oNMF decomposition for the current iteration exists
    if os.path.exists(save_path):
        print("oNMF decomposition with seed {} already exists. Skipping...".format(seed))
        return
    
    # Compute current oNMF decomposition and save the result
    current_onmf = compute_onmf(
        seed, num_onmf_components, gene_matrix_norm, max_iter=params['onmf_epoch_number'])
    np.save(save_path, current_onmf)
        

def subsample_normalize_gene_matrix(gene_matrix: np.ndarray,
                                    cluster_label: np.ndarray,
                                    num_subsample: int = 10000,
                                    seed: int = 0,
                                    params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample and balance the gene matrix based on desired sample size and standard deviation clipping.

        Parameters
        ----------
        gene_matrix : np.ndarray
            The gene expression matrix.
        cluster_label : np.ndarray
            Labels for clustering.
        num_subsample : int, optional
            Number of samples to subsample. Default is 10000.
        seed : int, optional
            Seed for reproducibility. Default is 0.
        params : dict, optional
            Additional parameters for subsampling and balancing. Default is None.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing the standard deviation of the subsampled matrix and the normalized gene matrix.
        """
        np.random.seed(seed)

        num_cell, num_gene = gene_matrix.shape

        method = params['balance_method']
        maximum_sample_rate = params['max_sample_rate']
        std_clip_percentile = params['std_clip_percentile']
        min_cluster_size = params['min_cluster_size']

        # If no balancing is required, simply subsample the matrix
        if method is None:
            gene_matrix_balanced = gene_matrix[np.random.choice(
                num_cell, num_subsample, replace=False), :]
        else:
            cluster_list, cluster_count = np.unique(cluster_label, return_counts=True)

            # Filter out clusters smaller than the minimum size
            if np.any(cluster_count < min_cluster_size):
                use_cluster = cluster_list[cluster_count >= min_cluster_size]
                filt_ind = np.isin(cluster_label, use_cluster)
                gene_matrix = gene_matrix[filt_ind, :]
                cluster_label = cluster_label[filt_ind]
                cluster_list, cluster_count = np.unique(cluster_label, return_counts=True)

                warnings.warn("Clusters smaller than {} cells are filtered out.".format(min_cluster_size))

            sampling_number = category_balance_number(num_subsample, cluster_count, method, maximum_sample_rate)

            gene_matrix_balanced = np.zeros(
                (np.sum(sampling_number), num_gene))

            for ii in range(len(cluster_list)):
                cur_num = sampling_number[ii]
                cur_filt = cluster_label == cluster_list[ii]
                sele_ind = np.random.choice(np.sum(cur_filt), cur_num)
                strart_ind = np.sum(sampling_number[:ii])
                end_ind = strart_ind + cur_num

                if issparse(gene_matrix):
                    gene_matrix_balanced[strart_ind: end_ind, :] = gene_matrix[cur_filt, :][sele_ind, :].toarray()
                else:
                    gene_matrix_balanced[strart_ind: end_ind, :] = gene_matrix[cur_filt, :][sele_ind, :]

        std = gene_matrix_balanced.std(axis=0)
        std_clipped = std.clip(np.percentile(std, std_clip_percentile), np.inf)
        gene_matrix_balanced_normalized = gene_matrix_balanced / std_clipped

        return std_clipped, gene_matrix_balanced_normalized


def corr_mean(cur_data: np.array) -> np.array:
    """
    Calculate the correlation matrix and mean of the given data.

    Parameters
    ----------
    cur_data : np.array
        Input data matrix.

    Returns
    -------
    np.array
        Array containing the correlation matrix and the mean of the data.
    """
    rec_data = np.zeros(2, dtype=object)
    rec_data[0] = cur_data.T.dot(cur_data) / cur_data.shape[0]
    rec_data[1] = np.mean(cur_data, axis=0).reshape(-1, 1)  

    return rec_data


def sample_corr_mean(samp_full: np.array, comp_bin: np.array) -> Tuple[np.array, np.array]:
    """
    Calculate the correlation mean for each unique sample in samp_full.

    Parameters
    ----------
    samp_full : np.array
        Array of samples.
    comp_bin : np.array
        Binary matrix representation of the samples.

    Returns
    -------
    Tuple[np.array, np.array]
        An array of correlation means for each unique sample.
        An array of unique sample labels.
    """
    samp_list = np.unique(samp_full)
    raw_corr_data = np.zeros(len(samp_list), dtype=object)

    # Calculating correlation mean for each unique sample
    for ind, samp in enumerate(samp_list):
        filt_ind = samp_full == samp
        raw_corr_data[ind] = corr_mean(comp_bin[filt_ind, :])

    return raw_corr_data, samp_list


def sample_states(samp_full: np.array, onmf_rep_tri: np.array) -> Tuple[np.array, np.array]:
    """
    Extract the state representations for each unique sample.

    Parameters
    ----------
    samp_full : np.array
        Array of sample labels.
    onmf_rep_tri : np.array
        Discretized ONMF representation.

    Returns
    -------
    Tuple[np.array, np.array]
        An array of state representations (transposed) for each unique sample.
        An array of unique sample labels.
    """
    samp_list = np.unique(samp_full)
    state_list = np.zeros(len(samp_list), dtype=object)

    for ii, cur_samp in enumerate(samp_list):
        cur_filt = samp_full == cur_samp
        cur_state = onmf_rep_tri[cur_filt, :]
        state_list[ii] = cur_state.T

    return state_list, samp_list


def para_moments_numpy(j_mat: np.array, h_vec: np.array) -> Tuple[np.array, np.array]:
    """
    Calculate the correlation and mean parameters given a j matrix and an h vector.

    Parameters
    ----------
    j_mat : np.array
        Interaction matrix.
    h_vec : np.array
        External field vector.

    Returns
    -------
    Tuple[np.array, np.array]
        The correlation parameter.
        The mean parameter.
    """
    num_spin = j_mat.shape[0]
    num_sample = 3 ** num_spin
    sample_indices = np.indices((3,) * num_spin)
    ordered_sample = (sample_indices - 1).reshape(num_spin, num_sample)

    # Adding diagonal elements to themselves for calculation
    j_mat = j_mat + np.diag(np.diag(j_mat))
    ordered_energy = - (h_vec.T @ ordered_sample + \
                        np.sum((j_mat @ ordered_sample) * ordered_sample, axis=0) / 2)
    ordered_exp = np.exp(-ordered_energy)
    partition = np.sum(ordered_exp)
    freq = ordered_exp / partition  # Calculating the frequency
    mean_para = np.sum(ordered_sample * freq.reshape(1, -1), axis=1)

    corr_para = np.einsum(
        'i,ji,ki->jk',
        freq.flatten(),
        ordered_sample,
        ordered_sample)

    return corr_para, mean_para

@numba.njit(fastmath=True, cache=True)
def para_moments(j_mat: np.ndarray,
                       h_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the correlation and mean parameters given an interaction
    matrix and an external field vector, using a Numba‑accelerated kernel.

    Parameters
    ----------
    j_mat : np.ndarray
        Interaction matrix.
    h_vec : np.ndarray
        External field vector.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The correlation parameter array of shape (n, n).
        The mean parameter array of shape (n,).
    """
    num_spin: int = j_mat.shape[0]

    # Copy so we can safely modify the diagonal in‑place
    j_dbl = j_mat.copy()
    for ii in range(num_spin):
        j_dbl[ii, ii] += j_mat[ii, ii]

    num_sample: int = 3 ** num_spin

    partition: float = 0.0
    mean_para = np.zeros(num_spin, dtype=np.float64)
    corr_para = np.zeros((num_spin, num_spin), dtype=np.float64)

    spins = np.empty(num_spin, dtype=np.int8)

    for idx in range(num_sample):
        # Decode `idx` → base‑3 spin configuration in {‑1, 0, 1}
        tmp = idx
        for s in range(num_spin - 1, -1, -1):
            spins[s] = (tmp % 3) - 1
            tmp //= 3

        # Energy: E = −h·s − ½ sᵀJs
        energy: float = 0.0
        for ii in range(num_spin):
            energy -= h_vec[ii] * spins[ii]

        js_term: float = 0.0
        for ii in range(num_spin):
            dot: float = 0.0
            for jj in range(num_spin):
                dot += j_dbl[ii, jj] * spins[jj]
            js_term += spins[ii] * dot
        energy -= 0.5 * js_term

        weight: float = np.exp(-energy)

        partition += weight
        for ii in range(num_spin):
            mean_para[ii] += spins[ii] * weight
            for jj in range(num_spin):
                corr_para[ii, jj] += spins[ii] * spins[jj] * weight

    mean_para /= partition
    corr_para /= partition

    return corr_para, mean_para


@numba.njit(fastmath=True)
def pseudol_gradient(cur_j: np.ndarray, 
                     cur_h: np.ndarray, 
                     cur_state: np.ndarray, 
                     directed: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the pseudo-likelihood gradients of j and h.

    Parameters
    ----------
    cur_j : np.ndarray
        Current j matrix.
    cur_h : np.ndarray
        Current h vector.
    cur_state : np.ndarray
        Current state matrix.
    directed : bool, optional
        Whether to compute directed gradients. Default is False.

    Returns
    -------
    Tuple[np.array, np.array]
        The gradient of j.
        The gradient of h.
    """
    num_spin, num_sample = cur_state.shape
    sample_factor = 1.0 / num_sample     

    diag_j = np.empty(num_spin, dtype=cur_j.dtype)
    for ii in range(num_spin):
        diag_j[ii] = cur_j[ii, ii]

    j_no_diag = cur_j.copy()
    for ii in range(num_spin):
        j_no_diag[ii, ii] = 0.0

    eff_h = j_no_diag @ cur_state + cur_h 

    exp_plus = np.empty_like(eff_h)
    exp_minus = np.empty_like(eff_h)
    denom = np.empty_like(eff_h)

    for ii in range(num_spin):
        exp_plus[ii, :] = np.exp(diag_j[ii] + eff_h[ii, :])
        exp_minus[ii, :] = np.exp(diag_j[ii] - eff_h[ii, :])
        denom[ii, :] = exp_plus[ii, :] + exp_minus[ii, :] + 1.0

    # Diagonal (self‑coupling) gradients 
    term_diag = cur_state * cur_state - (exp_plus + exp_minus) / denom

    # Off‑diagonal and field gradients 
    h_eff_grad = cur_state - (exp_plus - exp_minus) / denom
    cur_j_grad = (h_eff_grad @ cur_state.T) * sample_factor
    for ii in range(num_spin):
        cur_j_grad[ii, ii] = np.sum(term_diag[ii, :]) * sample_factor

    cur_h_grad = np.zeros((num_spin, 1), dtype=cur_j.dtype)
    for ii in range(num_spin):
        cur_h_grad[ii, 0] = np.sum(h_eff_grad[ii, :]) * sample_factor

    if not directed:
        cur_j_grad = 0.5 * (cur_j_grad + cur_j_grad.T)

    return - cur_j_grad, - cur_h_grad


@numba.njit()
def samp_moments(j_mat: np.ndarray, 
                 h_vec: np.ndarray, 
                 sample_size: int, 
                 mixing_time: int, 
                 samp_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample moments for the Markov Chain Monte Carlo (MCMC).

    Parameters
    ----------
    j_mat : np.ndarray
        Interaction matrix.
    h_vec : np.ndarray
        External field vector.
    sample_size : int
        Size of the sample.
    mixing_time : int
        Mixing time for the MCMC.
    samp_gap : int
        Gap between samples.

    Returns
    -------
    Tuple[np.array, np.array]
        The correlation parameter.
        The mean parameter.
    """
    per_batch = int(1e5)
    num_spin = j_mat.shape[0]
    rec_corr = np.zeros((num_spin, num_spin))
    rec_mean = np.zeros(num_spin)
    beta = 1
    batch_count = 1
    rec_sample = np.empty((num_spin, min(per_batch, int(sample_size))))
    cur_spin = (np.random.randint(0, 3, (num_spin, 1)) - 1).astype(np.float64)
    tot_sampling = int(
        mixing_time + sample_size * samp_gap - mixing_time % samp_gap)

    rand_ind = np.random.randint(0, num_spin, tot_sampling)
    rand_flip = np.random.randint(0, 2, tot_sampling)
    rand_prob = np.random.rand(tot_sampling)

    # Monte Carlo Sampling
    for ii in range(tot_sampling):
        cur_ind = rand_ind[ii]
        j_sub = j_mat[cur_ind, :]
        accept_prob = 0.0
        new_spin = 0.0
        diff_energy = 0.0

        if cur_spin[cur_ind] == 0:
            if rand_flip[ii] == 0:
                new_spin = 1.0
            else:
                new_spin = -1.0
            diff_energy = -j_mat[cur_ind, cur_ind] - new_spin * (j_sub.dot(cur_spin) + h_vec[cur_ind])
            accept_prob = min(1.0, np.exp(- diff_energy * beta)[0])
        else:
            if rand_flip[ii] == 0:
                accept_prob = 0;
            else:
                diff_energy = cur_spin[cur_ind] * (j_sub.dot(cur_spin) + h_vec[cur_ind])
                accept_prob = min(1.0, np.exp(- diff_energy * beta)[0])

        if rand_prob[ii] < accept_prob:
            if cur_spin[cur_ind] == 0:
                cur_spin[cur_ind] = new_spin
            else:
                cur_spin[cur_ind] = 0

        if ii > mixing_time:
            if (ii - mixing_time) % samp_gap == 0:
                rec_sample[:, batch_count - 1] = cur_spin[:, 0].copy()
                batch_count += 1

                if batch_count == per_batch + 1:
                    batch_count = 1
                    rec_sample = np.ascontiguousarray(rec_sample)
                    rec_corr += rec_sample.dot(rec_sample.T)
                    rec_mean += np.sum(rec_sample, axis=1)

    # Final processing of collected samples
    if batch_count != 1:
        cur_sample = rec_sample[:, :batch_count - 1]
        cur_sample = np.ascontiguousarray(cur_sample)
        rec_corr += cur_sample.dot(cur_sample.T)
        rec_mean += np.sum(cur_sample, axis=1)

    corr_para = rec_corr / sample_size
    mean_para = rec_mean / sample_size

    return corr_para, mean_para


def compute_gradient(cur_j: np.ndarray,
                            cur_h: np.ndarray,
                            raw_data: List[Tuple[np.ndarray, np.ndarray]],
                            method: str, train_dat: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient based on the specified method.

    Parameters
    ----------
    cur_j : np.ndarray
        Current j matrix.
    cur_h : np.ndarray
        Current h matrix.
    raw_data : List[Tuple[np.ndarray, np.ndarray]]
        Raw data used to calculate the gradient (each element is a tuple of correlation and mean).
    method : str
        The method to calculate the gradient; possible values are 'pseudo_likelihood',
        'maximum_likelihood', and 'mcmc_maximum_likelihood'.
    train_dat : Dict[str, Any]
        Training data and hyperparameters.

    Returns
    -------
    Tuple[np.array, np.array]
        The gradient of j.
        The gradient of h.
    """
    num_spin, num_round = cur_h.shape

    rec_jgrad = np.zeros((num_spin, num_spin, num_round))
    rec_hgrad = np.zeros((num_spin, num_round))

    for kk in range(num_round):
        # Computing gradients using the specified method.
        if method == 'pseudo_likelihood':
            j_grad, h_grad = pseudol_gradient(
                cur_j, cur_h[:, kk: kk + 1], raw_data[kk], directed=train_dat['directed'])
            h_grad = h_grad.flatten()
        else:
            # Distinguishing between other methods and computing gradients
            # accordingly
            if method == 'maximum_likelihood':
                corr_para, mean_para = para_moments(cur_j, cur_h[:, kk])
            elif method == 'mcmc_maximum_likelihood':
                corr_para, mean_para = samp_moments(
                    cur_j, cur_h[:, kk], train_dat['mcmc_samplingsz'], train_dat['mcmc_samplingmix'], train_dat['mcmc_samplegap'])
            j_grad = corr_para - raw_data[kk][0]
            h_grad = mean_para - raw_data[kk][1].flatten()

        rec_jgrad[:, :, kk] = j_grad
        rec_hgrad[:, kk] = h_grad

    return rec_jgrad, rec_hgrad


def _gradient_one_round(kk: int,
                        cur_j: np.ndarray,
                        cur_h: np.ndarray,
                        raw_data_k,
                        method: str,
                        train_dat: Dict[str, Any]):
    """Helper that compute gradient for a single round."""
    if method == 'pseudo_likelihood':
        j_grad, h_grad = pseudol_gradient(cur_j, cur_h[:, kk:kk + 1], raw_data_k, directed=train_dat['directed'])
        h_grad = h_grad.flatten()
    else:
        if method == 'maximum_likelihood':
            corr_para, mean_para = para_moments(cur_j, cur_h[:, kk])
        elif method == 'mcmc_maximum_likelihood':
            corr_para, mean_para = samp_moments(
                    cur_j, cur_h[:, kk], train_dat['mcmc_samplingsz'], train_dat['mcmc_samplingmix'], train_dat['mcmc_samplegap'])
        j_grad  = corr_para - raw_data_k[0]
        h_grad  = mean_para  - raw_data_k[1].flatten()

    return kk, j_grad, h_grad


def compute_gradient_para(cur_j: np.ndarray,
                     cur_h: np.ndarray,
                     raw_data: List[Tuple[np.ndarray, np.ndarray]],
                     method: str, train_dat: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient based on the specified method with parallel processing.

    Parameters
    ----------
    cur_j : np.ndarray
        Current j matrix.
    cur_h : np.ndarray
        Current h matrix.
    raw_data : List[Tuple[np.ndarray, np.ndarray]]
        Raw data used to calculate the gradient (each element is a tuple of correlation and mean).
    method : str
        The method to calculate the gradient; possible values are 'pseudo_likelihood',
        'maximum_likelihood', and 'mcmc_maximum_likelihood'.
    train_dat : Dict[str, Any]
        Training data and hyperparameters.

    Returns
    -------
    Tuple[np.array, np.array]
        The gradient of j.
        The gradient of h.
    """
    num_spin, num_round = cur_h.shape
    rec_jgrad = np.zeros((num_spin, num_spin, num_round))
    rec_hgrad = np.zeros((num_spin, num_round))

    n_jobs = train_dat.get('max_workers', min(num_round, os.cpu_count())) 
    chunk = train_dat.get('chunk_size', 1)          # rounds per task
    backend = 'loky'                                # process‑based

    # print(f'Processing with {n_jobs} workers...')
    with parallel_backend(backend, n_jobs=n_jobs):
        results = Parallel(batch_size=chunk)(
            delayed(_gradient_one_round)(kk, cur_j, cur_h, raw_data[kk], method, train_dat)
            for kk in range(num_round)
        )

    for kk, j_grad, h_grad in results:
        rec_jgrad[:, :, kk] = j_grad
        rec_hgrad[:, kk] = h_grad

    return rec_jgrad, rec_hgrad


def apply_regularization(rec_jgrad: np.ndarray, 
                         rec_hgrad: np.ndarray, 
                         cur_j: np.ndarray, 
                         cur_h: np.ndarray, 
                         train_dat: Dict[str, Any], 
                         print_regu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply regularization to the gradients.

    Parameters
    ----------
    rec_jgrad : np.ndarray
        Gradients of j matrix.
    rec_hgrad : np.ndarray
        Gradients of h matrix.
    cur_j : np.ndarray
        Current j matrix.
    cur_h : np.ndarray
        Current h matrix.
    train_dat : Dict[str, Any]
        Training data and hyperparameters.
    print_regu : bool, optional
        If True, print regularization parameters. Default is False.

    Returns
    -------
    Tuple[np.array, np.array]
        Regularized gradients of j.
        Regularized gradients of h.
    """
    num_spin, num_round = cur_h.shape

    lambda_l1_j, lambda_l1_h, lambda_l2_j, lambda_l2_h = (
        train_dat.get(
            key, 0) for key in [
            "lambda_l1_j", "lambda_l1_h", "lambda_l2_j", "lambda_l2_h"])

    if lambda_l1_j > 0:
        rec_jgrad += lambda_l1_j * \
            (cur_j / 1e-3).clip(-1, 1).reshape(num_spin, num_spin, 1)
    if lambda_l1_h > 0:
        rec_hgrad += lambda_l1_h * (cur_h / 1e-3).clip(-1, 1)

    if lambda_l2_j > 0:
        rec_jgrad += lambda_l2_j * cur_j.reshape(num_spin, num_spin, 1)
    if lambda_l2_h > 0:
        rec_hgrad += lambda_l2_h * cur_h

    if 'if_control' in train_dat:
        if_control = train_dat['if_control']
        batch_index = train_dat['batch_index']
        h_rela = compute_relative_responses(cur_h, if_control, batch_index)

        if 'lambda_l2_h_rela_prior' in train_dat:
            perturb_matrix = train_dat['perturb_matrix']
            rec_hgrad += train_dat['lambda_l2_h_rela_prior'] * (h_rela - perturb_matrix)

        if 'lambda_l1_h_rela' in train_dat:
            rec_hgrad += train_dat['lambda_l1_h_rela'] * (h_rela / 1e-3).clip(-1, 1)

        if 'lambda_l2_h_rela' in train_dat:
            rec_hgrad += train_dat['lambda_l2_h_rela'] * h_rela

    if 'lambda_l2_j_prior' in train_dat:
        rec_jgrad += (train_dat['lambda_l2_j_prior'] * (cur_j - train_dat['j_prior']))[:, :, np.newaxis]
    
    if 'lambda_l2_j_prior_mask' in train_dat:
        rec_jgrad += train_dat['lambda_l2_j_prior_mask'] * (cur_j * (train_dat['j_prior_mask'] == 0))[:, :, np.newaxis]
    
    if print_regu:  
        print('Regularization parameters')
        all_regu_list = ['lambda_l1_j', 'lambda_l1_h', 'lambda_l2_j', 'lambda_l2_h', 'lambda_l2_j_prior', 'lambda_l2_j_prior_mask', 'lambda_l2_h_rela_prior', 'lambda_l1_h_rela', 'lambda_l2_h_rela']
        all_values = [train_dat.get(key, 0) for key in all_regu_list]
        for key, value in zip(all_regu_list, all_values):
            if value > 0:
                print(f'{key}: {value}')


    return rec_jgrad, rec_hgrad


def update_adam(gradient: np.ndarray, 
                m: np.ndarray, 
                v: np.ndarray, 
                counter: int, 
                stepsz: float, 
                beta1: float = 0.9, 
                beta2: float = 0.999, 
                epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adam optimizer update rule.

    Parameters
    ----------
    gradient : np.ndarray
        The gradient of the objective function.
    m : np.ndarray
        1st moment vector (moving average of the gradients).
    v : np.ndarray
        2nd moment vector (moving average of the gradient squared).
    counter : int
        The current time step or epoch.
    stepsz : float
        Step size or learning rate.
    beta1 : float, optional
        The exponential decay rate for the 1st moment vector.
    beta2 : float, optional
        The exponential decay rate for the 2nd moment vector.
    epsilon : float, optional
        Small constant to prevent division by zero.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The update value.
        The updated first moment vector.
        The updated second moment vector.
    """
    # Update biased first moment estimate and biased second raw moment estimate
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)

    # Compute bias-corrected first moment estimate and bias-corrected second
    # raw moment estimate
    m_hat = m / (1 - beta1 ** counter)
    v_hat = v / (1 - beta2 ** counter)

    # Compute the update rule for parameters
    update = stepsz * m_hat / (np.sqrt(v_hat) + epsilon)

    return update, m, v


def learn_network_adam(raw_data: Any, 
                       method: str, 
                       train_dat: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train the network using the Adam optimizer.

    Parameters
    ----------
    raw_data : Any
        The input data.
    method : str
        The chosen method for training.
    train_dat : Dict[str, Any]
        Dictionary containing training data and hyperparameters.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        The trained j matrix.
        The trained h matrix.
        The training log.
    """
    np.random.seed(train_dat['seed'])

    # Retrieve training parameters and data
    num_spin, num_round = train_dat['cur_h'].shape
    num_epoch, stepsz, rec_gap = (
        train_dat.get(
            key, None) for key in [
            "num_epoch", "stepsz", "rec_gap"])
    list_step = np.arange(num_epoch, 0, - rec_gap)[::-1]
    cur_j, cur_h = (train_dat.get(key, None) for key in ["cur_j", "cur_h"])
    save_path = train_dat.get('save_path', None)
    backtrack_gap, backtrack_tol = (train_dat.get(key, None) for key in ["backtrack_gap", "backtrack_tol"])

    # Initialize variables to store parameters, gradients, and other values
    # during training
    rec_jmat_all = np.zeros((num_epoch, num_spin, num_spin))
    rec_hvec_all = np.zeros((num_epoch, num_spin, num_round))
    rec_jgrad_sum_norm = np.inf * np.ones(num_epoch)
    mjj, vjj = np.zeros(cur_j.shape), np.zeros(cur_j.shape)
    mhh, vhh = np.zeros(cur_h.shape), np.zeros(cur_h.shape)
    log_adam_grad = {name: deque(maxlen=backtrack_gap)
                     for name in ["mjj", "vjj", "mhh", "vhh"]}
    
    apply_regularization(np.zeros((num_spin, num_spin, num_round)), np.zeros((num_spin, num_round)), cur_j, cur_h, train_dat, print_regu=True)

    backtrack_counter = 0
    counter = 1
    
    tqdm._instances.clear()
    pbar = tqdm(total=num_epoch)

    while counter <= num_epoch:

        # Compute gradient and apply regularization
        rec_jgrad, rec_hgrad = compute_gradient(
            cur_j, cur_h, raw_data, method, train_dat)
        rec_jgrad, rec_hgrad = apply_regularization(
            rec_jgrad, rec_hgrad, cur_j, cur_h, train_dat, print_regu=False)

        # Update parameters using Adam optimizer
        rec_jgrad_sum = np.sum(rec_jgrad, axis=2)
        update, mjj, vjj = update_adam(
            rec_jgrad_sum, mjj, vjj, counter, stepsz)
        cur_j -= update
        update, mhh, vhh = update_adam(rec_hgrad, mhh, vhh, counter, stepsz)
        cur_h -= update

        # Store updated parameters and gradients for later analysis or use
        rec_jmat_all[counter - 1, :, :] = cur_j
        rec_hvec_all[counter - 1, :, :] = cur_h
        rec_jgrad_sum_norm[counter - 1] = np.linalg.norm(rec_jgrad_sum)

        # Log Adam gradients for backtracking
        for name, value in zip(["mjj", "vjj", "mhh", "vhh"], [
                               mjj, vjj, mhh, vhh]):
            log_adam_grad[name].append(value)

        # Save training log and print progress
        if counter in list_step:
            if train_dat['save_log']:
                sio.savemat(save_path + 'train_log.mat',
                            {'list_step': list_step,
                            'rec_hvec_all': rec_hvec_all,
                            'rec_jmat_all': rec_jmat_all,
                            'rec_jgrad_sum_norm': rec_jgrad_sum_norm})

            pbar.update(rec_gap)
            pbar.set_postfix({"Network Gradient": f"{rec_jgrad_sum_norm[counter - 1]:.6f}"})

        # Handle backtracking
        
        if_backtrack = False

        if (counter > backtrack_gap) and (rec_jgrad_sum_norm[counter - 1] > 1.5 * rec_jgrad_sum_norm[counter - 1 - backtrack_gap]):
            if_backtrack = True
            

        if (counter > 2 * backtrack_gap) and (counter % backtrack_gap == 0):

            est_cur_grad = rec_jgrad_sum_norm[counter - 1 - backtrack_gap: counter - 1].mean()
            est_prev_grad = rec_jgrad_sum_norm[counter - 1 - 2 * backtrack_gap: counter - 1 - backtrack_gap].mean()

            if est_cur_grad > 1.05 * est_prev_grad:
                if_backtrack = True


        if if_backtrack:
            print('Backtracking at epoch %d' % counter)
            backtrack_counter += 1
            mjj, vjj, mhh, vhh = [log_adam_grad[key][0]
                                  for key in ['mjj', 'vjj', 'mhh', 'vhh']]
            counter = counter - backtrack_gap
            stepsz = stepsz / 4
            if backtrack_counter > backtrack_tol:
                print(
                    'Backtracking more than %d times, stop training.' %
                    backtrack_tol)
                break
        else:
            counter += 1

    pbar.close()

    # Retrieve parameters corresponding to the minimum gradient norm found
    # during training
    trace_epoch = 50 + num_epoch - counter
    pos = num_epoch - trace_epoch + np.argmin(rec_jgrad_sum_norm[- trace_epoch: ])
    # print(pos)
    cur_h = rec_hvec_all[pos, :, :]
    cur_j = rec_jmat_all[pos, :, :]

    train_log = {}
    train_log['network_gradient'] = rec_jgrad_sum_norm[:counter]

    return cur_j, cur_h, train_log


def learn_program_regulators(gene_states: List[np.ndarray],
                             program_states: List[np.ndarray],
                             train_dat: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discover regulators for gene programs via regression.
    
    Parameters
    ----------
    gene_states : List[np.ndarray]
        Array of gene states. 
    program_states : List[np.ndarray]
        Array of program states.
    train_dat : Dict[str, Any]
        Dictionary of training parameters. 
          
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The learned interaction matrix of shape (n_spin, n_target).
        The learned self-interaction (regulator-specific bias) of shape (n_target,).
        The learned program activity bias of shape (n_target,).
    """
    num_gene = gene_states[0].shape[0]
    num_program = program_states[0].shape[0]
    num_round = len(gene_states)

    num_epoch, stepsz, rec_gap = (
        train_dat.get(
            key, None) for key in [
            "num_epoch", "stepsz", "rec_gap"])
    list_step = np.arange(num_epoch, 0, - rec_gap)[::-1]

    save_path = train_dat.get('save_path', None)

    backtrack_gap, backtrack_tol = (
        train_dat.get(
            key, None) for key in [
            "backtrack_gap", "backtrack_tol"])

    lambda_l1_j, lambda_l2_j = (train_dat.get(key, 0) for key in ["lambda_l1_j", "lambda_l2_j"])

    # Initialize parameters (using zeros)
    cur_interaction = np.zeros((num_gene, num_program))
    cur_selfj = np.zeros(num_program)
    cur_selfh = np.zeros(num_program)
    
    # Compute sample weights based on the number of states (columns) in each gene_states sample
    state_sizes = np.array([state.shape[1] for state in gene_states])
    samp_weight = np.sqrt(state_sizes)
    samp_weight = samp_weight / np.sum(samp_weight)
    
    # Preallocate gradient norm recording (optional)
    rec_grad = np.zeros(num_epoch)

    m_interaction, v_interaction = np.zeros_like(cur_interaction), np.zeros_like(cur_interaction)
    m_selfj, v_selfj = np.zeros_like(cur_selfj), np.zeros_like(cur_selfj)
    m_selfh, v_selfh = np.zeros_like(cur_selfh), np.zeros_like(cur_selfh)
    
    # Create l1 regularization schedule
    l1_base_start = 0.02
    l1_base_end = 1e-3
    
    part1 = np.full((num_epoch // 4 + 1,), l1_base_start)
    part2 = np.logspace(np.log10(l1_base_start), np.log10(l1_base_end), num=num_epoch // 2)
    part3 = np.full((num_epoch // 4 + 1,), l1_base_end)
    l1_base_array = np.concatenate([part1, part2, part3])[:num_epoch]
    
    # Preallocate arrays to collect per-sample gradients
    rec_interaction_grad = np.zeros((num_round, num_gene, num_program))
    rec_selfj_grad = np.zeros((num_round, num_program))
    rec_selfh_grad = np.zeros((num_round, num_program))
    
    pbar = tqdm(total=num_epoch)
    counter = 1

    while counter <= num_epoch:

        l1_base = l1_base_array[counter - 1]
        
        for kk in range(num_round):
            cur_state = gene_states[kk]         
            cur_target = program_states[kk]   
            
            effective_h = cur_state.T @ cur_interaction  + cur_selfh.T
            
            # j_sub is the bias for regulators (broadcasted as row vector)
            j_sub = cur_selfj.T  # shape: (n_target,)
            
            # Compute terms.
            # We follow MATLAB: term1 = exp(j_sub + effective_h)' so that term1 becomes (n_target, T)
            term1 = np.exp(j_sub + effective_h).T
            term2 = np.exp(j_sub - effective_h).T
            
            # Compute gradients for the bias parameters per sample
            j_sub_grad = cur_target**2 - (term1 + term2) / (term1 + term2 + 1)
            h_eff_grad = cur_target - (term1 - term2) / (term1 + term2 + 1)
            
            # Gradient for interaction parameters
            # cur_state: (n_spin, T), h_eff_grad.T: (T, n_target)
            j_off_sub_grad = cur_state @ h_eff_grad.T  # shape: (n_spin, n_target)
            
            # Average over the T (state) dimension
            rec_interaction_grad[kk] = j_off_sub_grad / cur_state.shape[1]
            rec_selfj_grad[kk] = np.mean(j_sub_grad, axis=1)
            rec_selfh_grad[kk] = np.mean(h_eff_grad, axis=1)
        
        # Aggregate gradients across samples using sample weights
        interaction_grad = - np.sum(rec_interaction_grad * samp_weight[:, None, None], axis=0)
        selfj_grad = - np.sum(rec_selfj_grad * samp_weight[:, None], axis=0)
        selfh_grad = - np.sum(rec_selfh_grad * samp_weight[:, None], axis=0)
        
        if lambda_l1_j > 0:
            interaction_grad += lambda_l1_j * (cur_interaction / l1_base).clip(- 1, 1)
        if lambda_l2_j > 0:
            interaction_grad += lambda_l2_j * cur_interaction

        # Update parameters using the provided update_adam function
        update_val, m_interaction, v_interaction = update_adam(interaction_grad, m_interaction, v_interaction, counter, stepsz)
        cur_interaction = cur_interaction - update_val
        
        update_val, m_selfj, v_selfj = update_adam(selfj_grad, m_selfj, v_selfj, counter, stepsz)
        cur_selfj = cur_selfj - update_val
        
        update_val, m_selfh, v_selfh = update_adam(selfh_grad, m_selfh, v_selfh, counter, stepsz)
        cur_selfh = cur_selfh - update_val
        
        # Record gradient norm (optional)
        rec_grad[counter - 1] = np.linalg.norm(interaction_grad)
        
        if counter in list_step:
            pbar.update(rec_gap)
            pbar.set_postfix({"Network Gradient": f"{rec_grad[counter - 1]:.4f}"})

        counter += 1
    
    return cur_interaction, cur_selfj, cur_selfh


def compute_relative_responses(cur_h: np.ndarray, 
                               if_control: np.ndarray, 
                               batch_index: np.ndarray) -> np.ndarray:
    """
    Compute the relative responses of samples by subtracting the average of control samples.
    This function calculates the relative responses for a set of samples by comparing each sample's response 
    to the average response of control samples. The control samples can be specific to the batch a sample belongs to, 
    or a global set of control samples if no control samples are present in the batch.

    Parameters
    ----------
    cur_h : np.ndarray
        A 2D array where each column represents a sample and each row represents a feature.
    if_control : np.ndarray
        A boolean array indicating which samples are control samples.
    batch_index : np.ndarray
        An array indicating the batch assignment for each sample.

    Returns
    -------
    np.ndarray
        A 2D array of the same shape as `cur_h`, containing the relative responses for each sample.
    """
    unique_batches = np.unique(batch_index)
    num_batches = len(unique_batches)

    relative_h = np.zeros_like(cur_h)
    
    if np.all(if_control == 0):
        all_controls_mean = np.mean(cur_h, axis=1)
    else:
        all_controls_mean = np.mean(cur_h[:, if_control], axis=1)
    
    for current_batch in unique_batches:
        batch_samples_idx = np.where(batch_index == current_batch)[0]

        if np.any(if_control[batch_samples_idx]):
            batch_controls_idx = batch_samples_idx[if_control[batch_samples_idx]]
            batch_controls_mean = np.mean(cur_h[:, batch_controls_idx], axis=1)
        else:
            batch_controls_mean = all_controls_mean
        
        for idx in batch_samples_idx:
            relative_h[:, idx] = cur_h[:, idx] - batch_controls_mean

    return relative_h


def select_representative_sample(raw_data: List[Tuple[np.ndarray, np.ndarray]], 
                                 num_select: int) -> List[int]:
    """
    Select representative samples by performing KMeans clustering on the samples'
    correlation and mean vectors and selecting the sample closest to the cluster center.
    
    Parameters
    ----------
    raw_data : List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples, each containing the correlation and mean vectors for a sample.
    num_select : int
        The number of representative samples to select.

    Returns
    -------
    List[int]
        A list of indices of the selected representative samples.
    """
    num_spin = raw_data[0][0].shape[1]
    num_samp = len(raw_data)
    raw_data_cat_vec = np.zeros([num_samp, int(num_spin * (num_spin + 3) / 2)])
    ind1, ind2 = np.triu_indices(num_spin)

    for ii in range(num_samp):
        cur_corrmean = raw_data[ii]
        triu_vect = cur_corrmean[0][ind1, ind2] / np.sqrt((num_spin + 1) / 2)
        triu_vect = np.concatenate([triu_vect, cur_corrmean[1].flatten()])
        raw_data_cat_vec[ii, :] = triu_vect

    pca_all = PCA().fit(raw_data_cat_vec)
    pca_rep = pca_all.transform(raw_data_cat_vec)
    kmeans = KMeans(n_clusters=num_select, n_init=200)
    kmeans.fit(raw_data_cat_vec)

    use_data_list = []
    for ii in range(num_select):
        cur_data = np.where(kmeans.labels_ == ii)[0]

        cur_center = kmeans.cluster_centers_[ii, :]
        data_dist = np.linalg.norm(raw_data_cat_vec[cur_data, :] - cur_center, axis=1)
        data_select = cur_data[np.argmin(data_dist)]

        use_data_list.append(data_select)
        
    sc.set_figure_params(figsize=[1, 1])
    fig, grid = sc.pl._tools._panel_grid(0.2, 0.2, ncols=3, num_panels=9)

    for ii in range(3):
        for jj in range(ii + 1):
            ax = plt.subplot(grid[ii * 3 + jj])
            plt.scatter(pca_rep[use_data_list, ii + 1], pca_rep[use_data_list, jj], s=90, color='#aaaaaa')
            plt.scatter(pca_rep[:, ii + 1], pca_rep[:, jj], s=20, c=kmeans.labels_, cmap='tab20', alpha=0.5)
            plt.xlabel('PC' + str(ii + 2))
            plt.ylabel('PC' + str(jj + 1))
            plt.xticks([])
            plt.yticks([])

    return use_data_list