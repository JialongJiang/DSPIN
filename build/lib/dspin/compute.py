# -*-coding:utf-8 -*-
'''
@Time    :   2023/09/14 21:51
@Author  :   Jialong Jiang, Yingying Gong
'''

from scipy.linalg import orth
from collections import deque
import numba
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import os
from sklearn.cluster import KMeans
import scipy.io as sio
from tqdm import tqdm
from typing import List


def category_balance_number(
        total_sample_size: int,
        cluster_count: List[int],
        method: str,
        maximum_sample_rate: float) -> np.array:
    '''
    Calculate the sampling number for each category based on the given method.

    Parameters:
    total_sample_size (int): The total size of the samples.
    cluster_count (List[int]): A list containing the count of elements in each cluster.
    method (str): The method used for balancing the categories; can be 'equal', 'proportional', or 'squareroot'.
    maximum_sample_rate (float): The maximum rate of the samples.

    Returns:
    np.array: An array containing the number of samples from each category based on the given method.
    '''

    # Validate the method parameter
    if method not in ['equal', 'proportional', 'squareroot']:
        raise ValueError(
            'method must be one of equal, proportional, squareroot')

    # Calculate sampling_number based on the given method
    if method == 'squareroot':
        # For squareroot method, calculate based on the square root of the
        # cluster count
        esti_size = (
            np.sqrt(cluster_count) /
            np.sum(
                np.sqrt(cluster_count)) *
            total_sample_size).astype(int)
        weight_fun = np.min(
            [esti_size, maximum_sample_rate * np.array(cluster_count)], axis=0)
    elif method == 'equal':
        # For equal method, divide total_sample_size by the number of clusters
        esti_size = total_sample_size / len(cluster_count)
        weight_fun = np.min([esti_size *
                             np.ones(len(cluster_count)), maximum_sample_rate *
                             np.array(cluster_count)], axis=0)
    else:
        # For proportional method, use cluster_count as weight_fun directly
        weight_fun = cluster_count

    # Calculate the final sampling number for each category
    sampling_number = (
        weight_fun /
        np.sum(weight_fun) *
        total_sample_size).astype(int)
    return sampling_number


def summary_components(all_components: np.array,
                       num_spin: int,
                       summary_method: str = 'kmeans') -> List[np.array]:
    '''
    Summarize components using KMeans clustering algorithm.

    Parameters:
    all_components (np.array): A 2D array where each row represents a sample, and each column represents a feature.
    num_spin (int): The number of clusters.
    summary_method (str, optional): The method used for summarizing the components. Defaults to 'kmeans'.

    Returns:
    List[np.array]: A list of numpy arrays, each containing the indices of the genes that belong to a specific group or cluster.
    '''

    # Number of genes (i.e., number of columns in all_components)
    num_gene = all_components.shape[1]

    if summary_method == 'kmeans':
        # Fit KMeans to components and their transpose
        kmeans = KMeans(
            n_clusters=num_spin,
            random_state=0,
            n_init=50).fit(all_components)
        kmeans_gene = KMeans(
            n_clusters=2 *
            num_spin,
            random_state=0,
            n_init=10).fit(
            all_components.T)

        # Initialize the array to store the average component for each cluster
        components_kmeans = np.zeros((num_spin, num_gene))

        for ii in range(num_spin):
            # Calculate the average component for each cluster
            components_kmeans[ii] = np.mean(
                all_components[kmeans.labels_ == ii], axis=0)

        # Normalize the components
        components_kmeans = normalize(components_kmeans, axis=1, norm='l2')

        gene_groups_ind = []
        for ii in range(num_spin):
            # Find which genes belong to which cluster
            gene_groups_ind.append(np.argmax(components_kmeans, axis=0) == ii)

    return gene_groups_ind


def onmf(X: np.array, rank: int, max_iter: int = 100) -> (np.array, np.array):
    """
    Orthogonal Non-Negative Matrix Factorization (ONMF) for a given rank.

    Parameters:
    X (np.array): Input Data Matrix.
    rank (int): Desired Rank for Factorization.
    max_iter (int, optional): Maximum Number of Iterations. Defaults to 100.

    Returns:
    np.array, np.array: Factorized Matrices S.T and A.
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
            pbar.set_postfix({"Reconstruction Error": f"{error:.2f}"})

    pbar.close()

    # Normalize the components
    norm_fac = np.sqrt(np.diag(S.dot(S.T)))
    S /= norm_fac.reshape(-1, 1)
    A *= norm_fac.reshape(1, -1)
    A *= np.sum(X * A.dot(S)) / np.sum((A.dot(S)) ** 2)

    return S.T, A


def compute_onmf(seed: int, num_spin: int, gene_matrix_bin: np.array) -> NMF:
    """
    Computes the ONMF model for the given gene matrix.

    Parameters:
    seed (int): Seed for Random Number Generation.
    num_spin (int): The number of desired components (clusters/spins).
    gene_matrix_bin (np.array): Binary Matrix representing the gene expression data.

    Returns:
    NMF: The NMF model with computed components.
    """

    # Generate a random seed
    np.random.seed(seed)
    # Factorized Matrices
    H, W = onmf(gene_matrix_bin, num_spin)

    # Initialize the NMF model
    nmf_model = NMF(n_components=num_spin, random_state=seed)

    # Set the components and number of components
    nmf_model.components_ = np.array(H).T
    nmf_model.n_components_ = num_spin

    return nmf_model


def onmf_discretize(onmf_rep_ori: np.array, fig_folder: str) -> np.array:
    """
    Discretize the representation obtained from ONMF using KMeans clustering and visualize the sorted representations.

    Parameters:
    onmf_rep_ori (np.array): Original ONMF representation.
    fig_folder (str): Folder to save the figure.

    Returns:
    np.array: Discretized ONMF representation.
    """

    num_spin = onmf_rep_ori.shape[1]
    sc.set_figure_params(figsize=[2, 2])
    _, grid = sc.pl._tools._panel_grid(
        0.3, 0.3, ncols=7, num_panels=min(
            21, num_spin))

    onmf_rep_tri = np.zeros(onmf_rep_ori.shape)
    for ii in range(num_spin):
        # Perform KMeans clustering for each spin/component
        km_fit = KMeans(n_clusters=3, n_init=10).fit(
            onmf_rep_ori[:, ii].reshape(-1, 1))
        onmf_rep_tri[:, ii] = (km_fit.labels_ == np.argsort(km_fit.cluster_centers_.reshape(-1))[0]) * \
            (-1) + (km_fit.labels_ == np.argsort(km_fit.cluster_centers_.reshape(-1))[2]) * 1

        # Visualize the sorted representations
        if ii < 21:
            ax = plt.subplot(grid[ii])
            plt.plot(np.sort(onmf_rep_ori[:, ii]))
            plt.plot(np.sort(km_fit.cluster_centers_[
                     km_fit.labels_].reshape(-1)))

    # Save the visual representation
    if fig_folder:
        plt.savefig(f'{fig_folder}onmf_discretize.png', bbox_inches='tight')
        plt.close()

    return onmf_rep_tri


def corr(data: np.array) -> np.array:
    """
    Calculates the correlation of the given data.

    Parameters:
    data (np.array): Input Data Matrix.

    Returns:
    np.array: Correlation Matrix of the given data.
    """

    # Transposing and dot multiplying the matrix, and normalizing by the
    # number of samples.
    return data.T.dot(data) / data.shape[0]


def corr_mean(cur_data: np.array) -> np.array:
    """
    Calculates the correlation and mean of the given data.

    Parameters:
    cur_data (np.array): Input Data Matrix.

    Returns:
    np.array: Array containing the correlation matrix and mean of the data.
    """

    rec_data = np.zeros(2, dtype=object)
    rec_data[0] = corr(cur_data)  # Storing correlation matrix
    rec_data[1] = np.mean(cur_data, axis=0).reshape(-1, 1)  # Storing mean

    return rec_data


def sample_corr_mean(samp_full: np.array,
                     comp_bin: np.array) -> (np.array,
                                             np.array):
    """
    Calculates the correlation mean for each unique sample in samp_full.

    Parameters:
    samp_full (np.array): Array of samples.
    comp_bin (np.array): Binary matrix representation of the samples.

    Returns:
    np.array, np.array: Array of correlation mean for each unique sample, and the array of unique samples.
    """

    samp_list = np.unique(samp_full)
    raw_corr_data = np.zeros(len(samp_list), dtype=object)

    # Calculating correlation mean for each unique sample
    for ind, samp in enumerate(samp_list):
        filt_ind = samp_full == samp
        raw_corr_data[ind] = corr_mean(comp_bin[filt_ind, :])

    return raw_corr_data, samp_list


def para_moments(j_mat: np.array, h_vec: np.array) -> (np.array, np.array):
    """
    Calculates the mean and correlation given j network and h vectors.

    Parameters:
    j_mat (np.array): Interaction Matrix.
    h_vec (np.array): External Field Vector.

    Returns:
    np.array, np.array: Correlation and mean parameters.
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


@numba.njit
def np_apply_along_axis(func1d, axis, arr):
    """
    Applies a function along the specified axis.

    Parameters:
    func1d: 1-D Function to be applied.
    axis (int): Axis along which function should be applied.
    arr (np.array): Input Array.

    Returns:
    np.array: The result of applying func1d to arr along the specified axis.
    """

    assert arr.ndim == 2
    assert axis in [0, 1]
    result = np.empty(arr.shape[1]) if axis == 0 else np.empty(arr.shape[0])

    for i in range(len(result)):
        result[i] = func1d(arr[:, i]) if axis == 0 else func1d(arr[i, :])

    return result


@numba.njit
def np_mean(array: np.array, axis: int) -> np.array:
    """
    Computes the arithmetic mean along the specified axis.

    Parameters:
    array (np.array): Input array.
    axis (int): Axis along which the mean is computed. 1 is for row-wise, 0 is for column-wise.
    """
    return np_apply_along_axis(np.mean, axis, array)


@numba.jit()
def pseudol_gradient(cur_j, cur_h, cur_state):
    """
    Compute the pseudo-likelihood gradients of j and h.

    Parameters:
    cur_j (numpy.ndarray): Current j matrix.
    cur_h (numpy.ndarray): Current h vector.
    cur_state (numpy.ndarray): Current state matrix.

    Returns:
    numpy.ndarray, numpy.ndarray: Gradients of j and h.
    """
    num_spin = cur_j.shape[0]

    cur_j_grad = np.zeros((num_spin, num_spin))
    cur_h_grad = np.zeros((num_spin, 1))

    # Filtering diagonal elements of cur_j matrix.
    j_filt = cur_j.copy()
    np.fill_diagonal(j_filt, 0)
    effective_h = j_filt.dot(cur_state) + cur_h

    for ii in range(num_spin):
        # Calculating gradients for each spin.
        j_sub = cur_j[ii, ii]
        h_sub = effective_h[ii, :]

        term1 = np.exp(j_sub + h_sub)
        term2 = np.exp(j_sub - h_sub)

        # Compute gradients.
        j_sub_grad = cur_state[ii, :] ** 2 - \
            (term1 + term2) / (term1 + term2 + 1)
        h_eff_grad = cur_state[ii, :] - (term1 - term2) / (term1 + term2 + 1)

        j_off_sub_grad = h_eff_grad * cur_state

        cur_j_grad[ii, :] = np_mean(j_off_sub_grad, axis=1)
        cur_j_grad[ii, ii] = np.mean(j_sub_grad)
        cur_h_grad[ii] = np.mean(h_eff_grad)

        # Symmetrize j gradient matrix
        cur_j_grad = (cur_j_grad + cur_j_grad.T) / 2

    return -cur_j_grad, -cur_h_grad


@numba.njit()
def samp_moments(j_mat, h_vec, sample_size, mixing_time, samp_gap):
    """
    Sample moments for the Markov Chain Monte Carlo (MCMC).

    Parameters:
    j_mat (numpy.ndarray): j matrix.
    h_vec (numpy.ndarray): h vector.
    sample_size (int): Size of the sample.
    mixing_time (int): Mixing time for the MCMC.
    samp_gap (int): Gap between samples.

    Returns:
    numpy.ndarray, numpy.ndarray: Correlation parameter and mean parameter.
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
        # Perform flips and accepts/rejects new states according to Metropolis
        # rule
        j_sub = j_mat[cur_ind, :]
        # Compute acceptance probability and execute possible spin flips
        # Collecting samples
        if ii > mixing_time and (ii - mixing_time) % samp_gap == 0:
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


def compute_gradient(cur_j, cur_h, raw_data, method, train_dat):
    """
    Compute the gradient based on the specified method.

    Parameters:
    cur_j (numpy.ndarray): Current j matrix.
    cur_h (numpy.ndarray): Current h matrix.
    raw_data (list): The raw data used to calculate the gradient.
    method (str): The method used to calculate the gradient. Possible values are 'pseudo_likelihood',
                  'maximum_likelihood', and 'mcmc_maximum_likelihood'.
    train_dat (dict): Training data information.

    Returns:
    numpy.ndarray, numpy.ndarray: Gradients of j and h.
    """
    num_spin, num_round = cur_h.shape

    rec_jgrad = np.zeros((num_spin, num_spin, num_round))
    rec_hgrad = np.zeros((num_spin, num_round))

    for kk in range(num_round):
        # Computing gradients using the specified method.
        if method == 'pseudo_likelihood':
            j_grad, h_grad = pseudol_gradient(
                cur_j, cur_h[:, kk].reshape(- 1, 1), raw_data[kk])
            h_grad = h_grad.flatten()
        else:
            # Distinguishing between other methods and computing gradients
            # accordingly
            if method == 'maximum_likelihood':
                corr_para, mean_para = para_moments(cur_j, cur_h[:, kk])
            elif method == 'mcmc_maximum_likelihood':
                corr_para, mean_para = samp_moments(
                    cur_j, cur_h, train_dat['mcmc_samplingsz'], train_dat['mcmc_samplingmix'], train_dat['mcmc_samplegap'])
            j_grad = corr_para - raw_data[kk][0]
            h_grad = mean_para - raw_data[kk][1].flatten()

        rec_jgrad[:, :, kk] = j_grad
        rec_hgrad[:, kk] = h_grad

    return rec_jgrad, rec_hgrad


def apply_regularization(rec_jgrad, rec_hgrad, cur_j, cur_h, train_dat):
    """
    Apply regularization to the gradients.

    Parameters:
    rec_jgrad (numpy.ndarray): Gradients of j matrix.
    rec_hgrad (numpy.ndarray): Gradients of h matrix.
    cur_j (numpy.ndarray): Current j matrix.
    cur_h (numpy.ndarray): Current h matrix.
    train_dat (dict): Training data information.

    Returns:
    numpy.ndarray, numpy.ndarray: Regularized gradients of j and h.
    """
    num_spin, num_round = cur_h.shape

    lambda_l1_j, lambda_l1_h, lambda_l2_j, lambda_l2_h, lambda_prior_h = (
        train_dat.get(
            key, 0) for key in [
            "lambda_l1_j", "lambda_l1_h", "lambda_l2_j", "lambda_l2_h", "lambda_prior_h"])

    if lambda_l1_j > 0:
        rec_jgrad += lambda_l1_j * \
            (cur_j / 0.02).clip(-1, 1).reshape(num_spin, num_spin, 1)
    if lambda_l1_h > 0:
        rec_hgrad += lambda_l1_h * (cur_h / 0.02).clip(-1, 1)

    if lambda_l2_j > 0:
        rec_jgrad += lambda_l2_j * cur_j
    if lambda_l2_h > 0:
        rec_hgrad += lambda_l2_h * cur_h

    return rec_jgrad, rec_hgrad


def update_adam(
        gradient,
        m,
        v,
        counter,
        stepsz,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8):
    """
    Adam optimizer update rule.

    Parameters:
    gradient (numpy.ndarray): The gradient of the objective function.
    m (numpy.ndarray): 1st moment vector (moving average of the gradients).
    v (numpy.ndarray): 2nd moment vector (moving average of the gradient squared).
    counter (int): The current time step or epoch.
    stepsz (float): Step size or learning rate.
    beta1 (float, optional): The exponential decay rate for the 1st moment vector.
    beta2 (float, optional): The exponential decay rate for the 2nd moment vector.
    epsilon (float, optional): Small constant to prevent division by zero.

    Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray: The updated parameters (update, m, v).
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


def learn_network_adam(raw_data, method, train_dat):
    """
    Train the network using Adam optimizer.

    Parameters:
    raw_data (object): The input data.
    method (str): The chosen method for training.
    train_dat (dict): Dictionary containing training data and hyperparameters.

    Returns:
    numpy.ndarray, numpy.ndarray: The trained network parameters (cur_j, cur_h).
    """
    # Retrieve training parameters and data
    num_spin, num_round = train_dat['cur_h'].shape
    num_epoch, stepsz, rec_gap = (
        train_dat.get(
            key, None) for key in [
            "num_epoch", "stepsz", "rec_gap"])
    list_step = np.arange(num_epoch, 0, - rec_gap)[::-1]
    cur_j, cur_h = (train_dat.get(key, None) for key in ["cur_j", "cur_h"])
    save_path = train_dat.get('save_path', None)
    backtrack_gap, backtrack_tol = (
        train_dat.get(
            key, None) for key in [
            "backtrack_gap", "backtrack_tol"])

    # Initialize variables to store parameters, gradients, and other values
    # during training
    rec_jmat_all = np.zeros((num_epoch, num_spin, num_spin))
    rec_hvec_all = np.zeros((num_epoch, num_spin, num_round))
    rec_jgrad_sum_norm = np.inf * np.ones(num_epoch)
    mjj, vjj = np.zeros(cur_j.shape), np.zeros(cur_j.shape)
    mhh, vhh = np.zeros(cur_h.shape), np.zeros(cur_h.shape)
    log_adam_grad = {name: deque(maxlen=backtrack_gap)
                     for name in ["mjj", "vjj", "mhh", "vhh"]}
    backtrack_counter = 0
    counter = 1

    while counter <= num_epoch:
        # Compute gradient and apply regularization
        rec_jgrad, rec_hgrad = compute_gradient(
            cur_j, cur_h, raw_data, method, train_dat)
        rec_jgrad, rec_hgrad = apply_regularization(
            rec_jgrad, rec_hgrad, cur_j, cur_h, train_dat)

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
            sio.savemat(save_path + 'train_log.mat',
                        {'list_step': list_step,
                         'rec_hvec_all': rec_hvec_all,
                         'rec_jmat_all': rec_jmat_all,
                         'rec_jgrad_sum_norm': rec_jgrad_sum_norm})
            print('Progress: %d, Network gradient: %f' % (
                np.round(100 * counter / num_epoch, 2), rec_jgrad_sum_norm[counter - 1]))

        # Handle backtracking
        if counter > backtrack_gap and rec_jgrad_sum_norm[counter - \
            1] > 2 * rec_jgrad_sum_norm[counter - 1 - backtrack_gap]:
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

    # Retrieve parameters corresponding to the minimum gradient norm found
    # during training
    pos = np.argmin(rec_jgrad_sum_norm)
    cur_h = rec_hvec_all[pos, :, :]
    cur_j = rec_jmat_all[pos, :, :]

    return cur_j, cur_h
