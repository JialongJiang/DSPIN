# -*-coding:utf-8 -*-
'''
@Time    :   2023/04/06 15:00
@Author  :   Jialong Jiang
'''

import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

import csv 

def onmf_to_csv(features, gene_name, data_folder, thres=0.01):

    num_spin = features.shape[0]
    rec_max_gene = 0
    with open(data_folder + 'onmf_gene_list_%d.csv' % num_spin,'w', newline='') as file:
        write = csv.writer(file)
        for spin in range(num_spin):
            num_gene_show = np.sum(features[spin, :] > thres)
            rec_max_gene = max(rec_max_gene, num_gene_show)
            gene_ind = np.argsort(- features[spin, :])[: num_gene_show]
            cur_line = list(gene_name[gene_ind])
            cur_line.insert(0, spin)
            write.writerow(cur_line) 

    pd_csv = pd.read_csv(data_folder + 'onmf_gene_list_%d.csv' % num_spin, names=range(rec_max_gene + 1))
    pd_csv = pd_csv.transpose()
    pd_csv.to_csv(data_folder + 'onmf_gene_list_%d.csv' % num_spin, header=False, index=False)
    return(data_folder + 'onmf_gene_list_%d.csv' % num_spin)

def onmf_gene_program_info(features, gene_name, num_gene_show, fig_folder=None):

    thres = 0.01
    num_spin = features.shape[0]
    sc.set_figure_params(figsize=[1.5, 3.6])
    fig, grid = sc.pl._tools._panel_grid(0.26, 0.9, ncols=6, num_panels=num_spin)

    for spin in range(num_spin):
        ax = plt.subplot(grid[spin])

        cur_num_gene_show = min(num_gene_show, np.sum(features[spin, :] > thres))
        gene_ind = np.argsort(- features[spin, :])[: cur_num_gene_show]
        plt.plot(features[spin, gene_ind], np.arange(cur_num_gene_show), 'o')
        plt.grid()
        plt.xlim([0, 1.1 * np.max(features[spin, gene_ind])])
        plt.ylim([-0.5, num_gene_show - 0.5])
        plt.yticks(np.arange(cur_num_gene_show), gene_name[gene_ind], fontsize=9);
        plt.gca().invert_yaxis()

        plt.title(spin)

    if fig_folder is not None:
        plt.savefig(fig_folder + 'onmf_gene_program_info.png', dpi=300, bbox_inches='tight')

    return(fig_folder + 'onmf_gene_program_info.png')

from scipy import optimize
from scipy.spatial.distance import pdist
def assign_program_position(onmf_rep_ori, umap_all, repulsion=2):

    num_spin = onmf_rep_ori.shape[1]
    program_umap_pos = np.zeros([num_spin, 2])
    for ii in range(num_spin):
        weight_sub = onmf_rep_ori[:, ii]
        weight_sub[weight_sub < np.percentile(weight_sub, 99.5)] = 0
        program_umap_pos[ii, :] = np.sum(umap_all * weight_sub.reshape(- 1, 1) / np.sum(weight_sub), axis=0)

    ori_pos = program_umap_pos.copy()
    def layout_loss_fun(xx):
        xx = xx.reshape(- 1, 2)
        attract = np.sum((xx - ori_pos) ** 2)
        repulse = np.sum(repulsion / pdist(xx))
        return attract + repulse

    opt_res = optimize.minimize(layout_loss_fun, program_umap_pos.flatten())
    program_umap_pos = opt_res.x.reshape(- 1, 2)

    sc.set_figure_params(figsize=[4, 4])

    
    if umap_all.shape[0] <= 5e4:
        plt.scatter(umap_all[:, 0], umap_all[:, 1], s=1, c='#bbbbbb', alpha=min(1, 1e4 / umap_all.shape[0]))
    else:
        sele_ind = np.random.choice(umap_all.shape[0], size=50000, replace=False).astype(int)
        plt.scatter(umap_all[sele_ind, 0], umap_all[sele_ind, 1], s=1, c='#bbbbbb', alpha=0.2)

    for ii in range(num_spin):
        plt.text(program_umap_pos[ii, 0], program_umap_pos[ii, 1], str(ii), fontsize=10)
    plt.axis('off')

    return program_umap_pos

import matplotlib.patheffects as PathEffects
def gene_program_on_umap(onmf_rep, umap_all, program_umap_pos, fig_folder=None, subsample=True):

    num_spin = onmf_rep.shape[1]

    if subsample:
        num_subsample = 20000
        sub_ind = np.random.choice(onmf_rep.shape[0], num_subsample, replace=False)
        onmf_rep = onmf_rep[sub_ind, :]
        umap_all = umap_all[sub_ind, :]

    sc.set_figure_params(figsize=[2, 2])
    fig, grid = sc.pl._tools._panel_grid(0.2, 0.06, ncols=6, num_panels=num_spin)
    for spin in range(num_spin):
        ax = plt.subplot(grid[spin])

        plt.scatter(umap_all[:, 0], umap_all[:, 1], c=onmf_rep[:, spin], s=1, 
        alpha=0.5, vmax=1, cmap='BuPu', vmin=-0.1)
        plt.text(program_umap_pos[spin, 0], program_umap_pos[spin, 1], str(spin), fontsize=12, path_effects=[PathEffects.withStroke(linewidth=3, foreground='w')])
        ax.set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.title(spin)

    if fig_folder is not None:
        plt.savefig(fig_folder + 'gene_program_on_umap.png', dpi=300, bbox_inches='tight')

    return(fig_folder + 'gene_program_on_umap.png')

def plot_j_network(j_mat, pos=None, label=None, thres=None, seed=0):
    
    if thres is not None:
        j_filt = j_mat.copy()
        j_filt[np.abs(j_mat) < thres] = 0
        np.fill_diagonal(j_filt, 0)
        j_mat = j_filt
        
    ax = plt.gca()
    
    nodesz = 0.2
    labeldist = 0.01
    linewz = 10

    G = nx.from_numpy_array(j_mat)

    eposi= [(u, v) for (u,v,d) in G.edges(data=True) if d['weight'] > 0]
    wposi= np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] > 0])

    enega = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] < 0]
    wnega = np.array([d['weight'] for (u,v,d) in G.edges(data=True) if d['weight'] < 0])
    
    if pos is None:
        pos = nx.spring_layout(G, weight=1, seed=seed)
        # pos = nx.spectral_layout(G)

    col1 = '#f0dab1'
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=61.8 * nodesz, node_color=col1, edgecolors=None)
    if label is not None:
        nx.draw_networkx_labels(G, pos, labels=label, font_size=20)
    
    sig_fun = lambda xx : (1 / (1 + np.exp(- 5 * (xx + cc))))
    cc = - np.max(np.abs(j_mat)) / 4
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=eposi, width=linewz * wposi, 
                           edge_color='#9999ff', alpha=sig_fun(wposi))

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=enega, width=- linewz * wnega, 
                           edge_color='#ff9999', alpha=sig_fun(- wnega))
    ax.set_axis_off()
        
    return pos
