# -*-coding:utf-8 -*-
'''
@Time    :   2023/03/22 17:20
@Author  :   Jialong Jiang
'''

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc 
from tqdm import tqdm

def corr(data):
    return data.T.dot(data) / data.shape[0]

def corr_mean(cur_data):
    rec_data = np.zeros(2, dtype=object)
    rec_data[0] = corr(cur_data)
    # np.fill_diagonal(rec_data[0], 1)
    rec_data[1] = np.mean(cur_data, axis=0).reshape(- 1, 1)
            
    return rec_data

def sample_corr_mean(samp_full, comp_bin):
    
    samp_list = np.unique(samp_full)
    raw_corr_data = np.zeros(len(samp_list), dtype=object)
    comp_bin = comp_bin * 2 - 1
    
    for ind, samp in enumerate(samp_list):
        filt_ind = samp_full == samp
        raw_corr_data[ind] = corr_mean(comp_bin[filt_ind, :])
        
    return raw_corr_data, samp_list

def wthresh(x, thresh):
    """
    Perform soft thresholding on an input array x, with a given threshold value.
    """
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

def para_moments(j_mat, h_vec):
    """
    Calculate the mean and correlation given j network and h vectors"""
    num_spin = j_mat.shape[0]
    num_sample = 3 ** num_spin

    sample_indices = np.indices((3,) * num_spin)
    ordered_sample = (sample_indices - 1).reshape(num_spin, num_sample)

    # Calculate ordered energy and partition function
    j_mat = j_mat + np.diag(np.diag(j_mat))
    ordered_energy = - (h_vec.T @ ordered_sample + np.sum((j_mat @ ordered_sample) * ordered_sample, axis=0) / 2)

    ordered_exp = np.exp(-ordered_energy)
    partition = np.sum(ordered_exp)
    freq = ordered_exp / partition

    mean_para = np.sum(ordered_sample * freq.reshape(1, - 1), axis=1)

    corr_para = np.einsum('i,ji,ki->jk', freq.flatten(), ordered_sample, ordered_sample)

    return corr_para, mean_para


import numba

@numba.njit()
def samp_moments(j_mat, h_vec, sample_size, mixing_time, samp_gap):
    per_batch = int(1e6)
    
    num_spin = j_mat.shape[0]
    rec_corr = np.zeros((num_spin, num_spin))
    rec_mean = np.zeros(num_spin)
    beta = 1
    batch_count = 1
    rec_sample = np.empty((num_spin, min(per_batch, int(sample_size))))
    cur_spin = (np.random.randint(0, 3, (num_spin, 1)) - 1).astype(np.float64)
    tot_sampling = int(mixing_time + sample_size * samp_gap - mixing_time % samp_gap)

    rand_ind = np.random.randint(0, num_spin, tot_sampling)
    rand_flip = np.random.randint(0, 2, tot_sampling)
    rand_prob = np.random.rand(tot_sampling)

    # for ii in numba.prange(tot_sampling):
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

    if batch_count != 1:
        cur_sample = rec_sample[:, :batch_count - 1]
        cur_sample = np.ascontiguousarray(cur_sample)
        rec_corr += cur_sample.dot(cur_sample.T)
        rec_mean += np.sum(cur_sample, axis=1)

    corr_para = rec_corr / sample_size
    mean_para = rec_mean / sample_size

    return corr_para, mean_para

import scipy.io as sio
def learn_jmat_adam(corrs, means, train_dat):

    num_round = corrs.shape[2]
    num_spin = corrs.shape[0]

    num_epoch = train_dat["epoch"]
    stepsz = train_dat["stepsz"]
    counter = train_dat["counter"]
    task_name = train_dat["task_name"]
    samplingsz_raw = train_dat["samplingsz"]
    samplingmix = train_dat["samplingmix"]
    rec_gap = train_dat["rec_gap"]
    spin_thres = train_dat["spin_thres"]
    samplingsz_step = samplingsz_raw / num_epoch * 2

    list_step = np.arange(num_epoch, 0, -rec_gap)[::-1]
    num_rec = len(list_step)

    cur_j = train_dat["cur_j"]
    cur_h = train_dat["cur_h"]

    rec_jgrad = np.zeros((num_spin, num_spin, num_round))
    rec_hgrad = np.zeros((num_spin, num_round))

    rec_hgrad_norm = np.zeros((num_round, num_epoch))
    rec_hvec_all = np.zeros((num_spin, num_round, num_epoch))
    rec_jgrad_norm = np.zeros((num_round, num_epoch))
    rec_jgrad_sum_norm = np.zeros((num_epoch, 1))
    rec_jmat_all = np.zeros((num_spin, num_spin, num_epoch))

    count = 1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-6
    mjj = np.zeros((num_spin,))
    vjj = np.zeros((num_spin,))
    mhh = np.zeros((num_spin, num_round))
    vhh = np.zeros((num_spin, num_round))
    mjj_log = np.zeros((num_spin, num_spin, num_epoch))
    vjj_log = np.zeros((num_spin, num_spin, num_epoch))
    mhh_log = np.zeros((num_spin, num_round, num_epoch))
    vhh_log = np.zeros((num_spin, num_round, num_epoch))

    jj = 1
    while jj <= num_epoch:

        samplingsz = round(min(jj * samplingsz_step, samplingsz_raw))
        # samplingsz = samplingsz_raw

        rec_jgrad = np.zeros((num_spin, num_spin, num_round))
        rec_hgrad = np.zeros((num_spin, num_round))
        for kk in range(1, num_round+1):
            if num_spin <= spin_thres:
                corr_para, mean_para = para_moments(cur_j, cur_h[:, kk-1])
            else:
                corr_para, mean_para = samp_moments(cur_j, cur_h[:, kk-1], samplingsz, samplingmix, 1)
            rec_jgrad[:, :, kk-1] = (corr_para - corrs[:, :, kk-1])
            rec_hgrad[:, kk-1] = (mean_para - means[:, kk-1])

        if 'lam_l1j' in train_dat:
            rec_jgrad = rec_jgrad + train_dat['lam_l1j'] * np.sign(wthresh(cur_j, 1e-3)).reshape((num_spin, num_spin, 1))
        if 'lam_l2j' in train_dat:
            rec_jgrad = rec_jgrad + train_dat['lam_l2j'] * cur_j.reshape((num_spin, num_spin, 1))

        rec_jgrad_full = rec_jgrad
        rec_jgrad = np.sum(rec_jgrad, axis=2) / num_round
      


        mjj = beta1 * mjj + (1 - beta1) * rec_jgrad
        vjj = beta2 * vjj + (1 - beta2) * (rec_jgrad ** 2)

        mHatjj = mjj / (1 - beta1 ** counter)
        vHatjj = vjj / (1 - beta2 ** counter)

        vfStepjj = stepsz * mHatjj / (np.sqrt(vHatjj) + epsilon)
        cur_j = cur_j - vfStepjj

        if 'lam_l2h' in train_dat:
            rec_hgrad = rec_hgrad + train_dat['lam_l2h'] * cur_h

        rec_hgrad_full = rec_hgrad
        mhh = beta1 * mhh + (1 - beta1) * rec_hgrad
        vhh = beta2 * vhh + (1 - beta2) * (rec_hgrad ** 2)

        mHathh = mhh / (1 - beta1 ** counter)
        vHathh = vhh / (1 - beta2 ** counter)

        vfStephh = stepsz * mHathh / (np.sqrt(vHathh) + epsilon)
        cur_h = cur_h - vfStephh

        rec_hvec_all[:, :, jj-1] = cur_h
        rec_jmat_all[:, :, jj-1] = cur_j

        rec_hgrad_norm[:, jj-1] = np.sqrt(np.sum(rec_hgrad_full ** 2, axis=0))
        rec_jgrad_norm[:, jj-1] = np.sqrt(np.sum(rec_jgrad_full ** 2, axis=(0, 1)))


        rec_jgrad_sum_norm[jj - 1] = np.sqrt(np.sum(np.sum(rec_jgrad_full, axis=2) ** 2, axis=(0, 1)))

        mjj_log[:, :, jj-1] = mjj 
        vjj_log[:, :, jj-1] = vjj
        mhh_log[:, :, jj-1] = mhh 
        vhh_log[:, :, jj-1] = vhh 
    
        if jj == list_step[count]:
            

            sio.savemat(task_name + '.mat', {'count':count, 'list_step':list_step, 
                                            'rec_hvec_all':rec_hvec_all, 
                                            'rec_jmat_all':rec_jmat_all, 
                                            'rec_hgrad_norm':rec_hgrad_norm, 
                                            'rec_jgrad_norm':rec_jgrad_norm, 
                                            'rec_jgrad_sum_norm':rec_jgrad_sum_norm})
                                    
            print('Progress: %d, Network gradient: %f' %(np.round(100 * jj / num_epoch, 2), rec_jgrad_sum_norm[jj-1]))
            count += 1
          

        step_gap = 20
        if jj > (step_gap) and rec_jgrad_sum_norm[jj - 1] > 2 * rec_jgrad_sum_norm[jj - 1 - step_gap]:
            print('warning: backtrack')
            mjj = mjj_log[:, :, jj-1 - step_gap] 
            vjj = vjj_log[:, :, jj-1 - step_gap] 
            mhh = mhh_log[:, :, jj-1 - step_gap] 
            vhh = vhh_log[:, :, jj-1 - step_gap] 
            jj = jj - step_gap 
            counter = counter - step_gap
            stepsz = stepsz / 2
        else:
            counter = counter + 1
            jj = jj + 1

    pos = np.argmin(rec_jgrad_sum_norm)
    cur_h = rec_hvec_all[:, :, pos]
    cur_j = rec_jmat_all[:, :, pos]

    return cur_j, cur_h

from scipy.linalg import orth 
from sklearn.decomposition import NMF

def onmf(X, rank, max_iter=500):
    
    m, n = X.shape
    
    A = np.random.rand(m, rank) 
    S = np.random.rand(rank, n)
    S = np.abs(orth(S.T).T)
    
    for itr in tqdm(range(max_iter)):
            
        coef_A = X.dot(S.T) / A.dot(S.dot(S.T))
        A = np.nan_to_num(A * coef_A) #, posinf=1e5)

        AtX = A.T.dot(X)
        coef_S = AtX / S.dot(AtX.T).dot(S)
        S = np.nan_to_num(S * coef_S) #, posinf=1e5)

    norm_fac = np.sqrt(np.diag(S.dot(S.T)))
    S = S / norm_fac.reshape(- 1, 1)
    A = A * norm_fac.reshape(1, - 1)
    
    as_prod = A.dot(S)
    const = np.sum(X * as_prod) / np.sum(as_prod ** 2)
    A *= const

    return S.T, A

def compute_onmf(seed, num_spin, gene_matrix_bin):
    np.random.seed(seed)
    H, W = onmf(gene_matrix_bin, num_spin)

    nmf_model = NMF(n_components=num_spin, random_state=seed)
    nmf_model.components_ = np.array(H).T
    nmf_model.n_components_ = num_spin
    return nmf_model

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def summarize_onmf_decomposition(num_spin, num_repeat, num_pool, onmf_path, gene_matrix, fig_folder=None):

    rec_components = np.zeros((num_repeat, num_pool, gene_matrix.shape[1]))

    for ii in range(num_repeat):
        cur_onmf = np.load('%sonmf_%d_%d.npy' % (onmf_path, num_pool, ii + 1), allow_pickle=True).item()
        rec_components[ii] = cur_onmf.components_

    all_components = rec_components.reshape(num_repeat * num_pool, -1)

    gene_weight = np.sum(all_components ** 2, axis=0) ** 0.5
    gene_sele_filt = gene_weight > np.mean(gene_weight)
    all_components_sub = all_components[:, gene_sele_filt]

    kmeans = KMeans(n_clusters=num_spin, random_state=0).fit(all_components)
    kmeans_gene = KMeans(n_clusters=num_spin, random_state=0).fit(all_components_sub.T)

    components_kmeans = np.zeros((num_spin, gene_matrix.shape[1]))
    for ii in range(num_spin):
        components_kmeans[ii] = np.mean(all_components[kmeans.labels_ == ii], axis=0)
    components_kmeans = normalize(components_kmeans, axis=1, norm='l2')
    
    components_summary = np.zeros((num_spin, gene_matrix.shape[1]))
    for ii in range(num_spin):
        filt_genes = np.argmax(components_kmeans, axis=0) == ii
        # filt_genes = components_kmeans[ii] > 1e-3
        sub_matrix = gene_matrix[:, filt_genes]

        sub_onmf = NMF(n_components=1, init='random', random_state=0).fit(sub_matrix)
        components_summary[ii, filt_genes] = sub_onmf.components_[0]
    components_summary = normalize(components_summary, axis=1, norm='l2')
    
    # components_summary = components_kmeans

    onmf_summary = NMF(n_components=num_spin, init='random', random_state=0)
    onmf_summary.components_ = components_summary

    sc.set_figure_params(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    gene_order = np.argsort(kmeans_gene.labels_)
    comp_order = np.argsort(kmeans.labels_)
    plt.imshow(all_components_sub[comp_order, :][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(all_components) / 10, interpolation='none')
    plt.title('All components')

    plt.subplot(1, 3, 2)
    plt.imshow(components_kmeans[:, gene_sele_filt][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(components_kmeans) / 10, interpolation='none')
    plt.title('Kmeans components')

    plt.subplot(1, 3, 3)
    plt.imshow(components_summary[:, gene_sele_filt][:, gene_order], aspect='auto', cmap='Blues', vmax=np.max(all_components) / 10, interpolation='none')
    plt.title('Summary components')

    if fig_folder is not None:
        plt.savefig(fig_folder + 'onmf_decomposition_summary.png', dpi=300, bbox_inches='tight')

    return onmf_summary

def onmf_discretize(onmf_rep_ori, fig_folder):

    num_spin = onmf_rep_ori.shape[1]
    sc.set_figure_params(figsize=[2, 2])
    fig, grid = sc.pl._tools._panel_grid(0.3, 0.3, ncols=7, num_panels=num_spin)
    onmf_rep_tri = np.zeros(onmf_rep_ori.shape) 
    rec_kmeans = np.zeros(num_spin, dtype=object)
    for ii in tqdm(range(num_spin)):
        ax = plt.subplot(grid[ii])
        
        km_fit = KMeans(n_clusters=3, n_init=10)
        km_fit.fit(onmf_rep_ori[:, ii].reshape(- 1, 1))
        plt.plot(np.sort(onmf_rep_ori[:, ii]));
        plt.plot(np.sort(km_fit.cluster_centers_[km_fit.labels_].reshape(- 1)));
        
        label_ord = np.argsort(km_fit.cluster_centers_.reshape(- 1))
        onmf_rep_tri[:, ii] = (km_fit.labels_ == label_ord[1]) * 0.5 + (km_fit.labels_ == label_ord[2]) * 1
        rec_kmeans[ii] = km_fit

    if fig_folder is not None:
        plt.savefig(fig_folder + 'onmf_discretize.png', dpi=300, bbox_inches='tight')
        
    return onmf_rep_tri, rec_kmeans

from scipy.sparse import issparse
def prepare_onmf_decomposition(cadata, data_folder, balance_by='leiden', total_sample_size=1e5, method='squareroot'):
    
    if issparse(cadata.X):
        cadata.X = cadata.X.toarray()

    maximum_sample_rate = 2
    cluster_list = list(cadata.obs[balance_by].value_counts().keys())
    cluster_count = list(cadata.obs[balance_by].value_counts())

    if method == 'porpotional':
        weight_fun = cluster_count
    elif method == 'squareroot':
        esti_size = (np.sqrt(cluster_count) / np.sum(np.sqrt(cluster_count)) * total_sample_size).astype(int)
        weight_fun = np.min([esti_size, maximum_sample_rate * np.array(cluster_count)], axis=0)
    elif method == 'equal':
        esti_size = total_sample_size / len(cluster_list)
        weight_fun = np.min([esti_size * np.ones(len(cluster_count)), maximum_sample_rate * np.array(cluster_count)], axis=0)
    sampling_number = (weight_fun / np.sum(weight_fun) * total_sample_size).astype(int)

    gene_matrix_balanced = np.zeros((np.sum(sampling_number), cadata.X.shape[1]))

    for ii in tqdm(range(len(cluster_list))):
        cur_num = sampling_number[ii]
        cur_filt = cadata.obs[balance_by] == cluster_list[ii]
        sele_ind = np.random.choice(np.sum(cur_filt), cur_num)
        strart_ind = np.sum(sampling_number[:ii])
        end_ind = strart_ind + cur_num
        gene_matrix_balanced[strart_ind: end_ind, :] = cadata.X[cur_filt, :][sele_ind, :]

    matrix_path = data_folder + 'gmatrix_' + '{:.0e}'.format(total_sample_size) + '_balanced_' + method + '.npy'
    np.save(matrix_path, gene_matrix_balanced)

    return matrix_path

def select_diverse_sample(raw_data_tri, num_cluster, fig_folder):

    num_spin = raw_data_tri[0][0].shape[1]
    num_samp = len(raw_data_tri)
    raw_data_cat_vec = np.zeros([num_samp, int(num_spin * (num_spin + 3) / 2)])
    ind1, ind2 = np.triu_indices(num_spin)

    for ii in range(num_samp):
        cur_corrmean = raw_data_tri[ii]
        triu_vect = cur_corrmean[0][ind1, ind2] / np.sqrt((num_spin + 1) / 2)
        triu_vect = np.concatenate([triu_vect, cur_corrmean[1].flatten()])
        raw_data_cat_vec[ii, :] = triu_vect

    from sklearn.decomposition import PCA

    pca_all = PCA().fit(raw_data_cat_vec)
    pca_rep = pca_all.transform(raw_data_cat_vec)
    kmeans = KMeans(n_clusters=num_cluster, n_init=200)
    kmeans.fit(raw_data_cat_vec)

    use_data_list = []
    for ii in range(num_cluster):
        cur_data = np.where(kmeans.labels_ == ii)[0]

        cur_center = kmeans.cluster_centers_[ii, :]
        data_dist = np.linalg.norm(raw_data_cat_vec[cur_data, :] - cur_center, axis=1)
        data_select = cur_data[np.argmin(data_dist)]

        use_data_list.append(data_select)
    
    
    sc.set_figure_params(figsize=[3, 3])
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

    plt.savefig(fig_folder + 'pca_cluster.png', dpi=300, bbox_inches='tight')

    return use_data_list