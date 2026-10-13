# -*-coding:utf-8 -*-
'''
@Time    :   2026/07/30
@Author  :   Jialong Jiang
'''

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.optimize import curve_fit
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression
from typing import Optional


def select_highly_variable_genes(
        adata: ad.AnnData, 
        num_gene_select: int = 3000,
        detect_frac: float = 0.02, 
        max_mean_percentile: float = 100,
        fit_method: str = 'nb',
        group_key: Optional[str] = None) -> np.ndarray:
    """
    Select highly variable genes using a fitted mean-CV model.

    The input matrix is expected to contain normalized, non-log-transformed
    expression values. Linear fit, negative-binomial fit, and selected highly
    variable gene plots are displayed.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix containing normalized gene expression.
    num_gene_select : int, optional
        Number of highly variable genes to select. Default is 3000.
    detect_frac : float, optional
        Minimum fraction of cells expressing a gene across all cells, or in at
        least one group when ``group_key`` is provided. Default is 0.02.
    max_mean_percentile : float, optional
        Upper percentile of gene means retained for selection. Values below
        100 remove highly expressed housekeeping genes. Default is 100.
    fit_method : str, optional
        Fitted baseline used to rank genes. Options are 'linear' or 'nb'.
        Default is 'nb'.
    group_key : str or None, optional
        Column in ``adata.obs`` used to group cells for the detection filter.
        If None, detection is calculated across all cells. Default is None.

    Returns
    -------
    np.ndarray
        Names of the selected highly variable genes.
    """
    batch_size = 2000
    sum_x = np.zeros(adata.n_vars)
    sum_x_squared = np.zeros(adata.n_vars)
    for start in range(0, adata.n_obs, batch_size):
        block = adata.X[start:start + batch_size]
        sum_x += np.asarray(block.sum(axis=0)).ravel()
        sum_x_squared += np.asarray((block.power(2) if issparse(block) else np.square(block)).sum(axis=0)).ravel()

    data_mean = sum_x / adata.n_obs
    data_var = sum_x_squared / adata.n_obs - data_mean ** 2
    valid = (data_mean > 0) & (data_var > 0)

    log_mean = np.full(adata.n_vars, np.nan)
    log_cv = np.full(adata.n_vars, np.nan)
    log_mean[valid] = np.log10(data_mean[valid])
    log_cv[valid] = np.log10(np.sqrt(data_var[valid]) / data_mean[valid])

    def nb_model(x, k, b, pi):
        return k * (x - np.log(1 + pi * np.exp(x))) + b

    linear_model = LinearRegression().fit(
        log_mean[valid, None], log_cv[valid])
    linear_fit = np.full(adata.n_vars, np.nan)
    linear_fit[valid] = linear_model.predict(log_mean[valid, None])

    popt, _ = curve_fit(
        nb_model, log_mean[valid], log_cv[valid], [-0.5, 1, 0],
        method='dogbox')
    nb_fit = np.full(adata.n_vars, np.nan)
    nb_fit[valid] = nb_model(log_mean[valid], *popt)

    if fit_method == 'linear':
        fit_res = linear_fit
    elif fit_method == 'nb':
        fit_res = nb_fit
    else:
        raise ValueError("fit_method must be one of 'linear' or 'nb'")

    if group_key is None:
        detected = np.asarray((adata.X > 0).mean(axis=0)).ravel()
    else:
        detected = np.zeros(adata.n_vars)
        groups = adata.obs[group_key].astype(str).to_numpy()
        for group in np.unique(groups):
            group_detect = np.asarray(
                (adata.X[groups == group] > 0).mean(axis=0)).ravel()
            detected = np.maximum(detected, group_detect)

    eligible = (detected > detect_frac) & valid
    if max_mean_percentile < 100:
        mean_cutoff = np.percentile(data_mean[valid], max_mean_percentile)
        eligible &= data_mean <= mean_cutoff

    score = np.full(adata.n_vars, -np.inf)
    score[valid] = log_cv[valid] - fit_res[valid]
    cutoff = np.sort(score[eligible])[::-1][num_gene_select - 1]
    gene_filt = (score >= cutoff) & eligible

    sc.set_figure_params(figsize=[3, 3])
    fig, grid = sc.pl._tools._panel_grid(0.1, 0.3, ncols=3, num_panels=3)
    ax = [plt.subplot(grid[ii]) for ii in range(3)]
    ax[0].scatter(log_mean[valid], log_cv[valid], 5)
    ax[0].scatter(log_mean[valid], linear_fit[valid], 5)
    ax[0].set(
        title=f'linear fit (k={linear_model.coef_[0]:.3f})',
        xlabel='log10(mean)', ylabel='log10(cv)')

    ax[1].scatter(log_mean[valid], log_cv[valid], 5)
    ax[1].scatter(log_mean[valid], nb_fit[valid], 5)
    ax[1].set(
        title=f'NB model (k={popt[0]:.3f})',
        xlabel='log10(mean)', ylabel='log10(cv)')
        
    ax[2].scatter(log_mean[valid], log_cv[valid], 1, alpha=0.5)
    ax[2].scatter(
        log_mean[gene_filt], log_cv[gene_filt], 5, alpha=0.5, label='HVG')
    ax[2].legend()
    ax[2].set(xlabel='log10(mean)', ylabel='log10(cv)')
    plt.show()

    return adata.var_names[gene_filt].to_numpy()
