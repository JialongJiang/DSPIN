# -*-coding:utf-8 -*-
'''
@Time    :   2023/10/30 15:41
@Author  :   Jialong Jiang, Yingying Gong
'''

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import csv
import scanpy as sc
import igraph as ig
import leidenalg as la
import networkx as nx
from scipy import optimize
from scipy.spatial.distance import pdist


def onmf_to_csv(features, gene_name, file_path, thres=0.01):
    """
    Extract gene names in ONMF to CSV.

    Parameters:
    features (numpy.ndarray): oNMF features to be written to CSV.
    gene_name (List[str]): Names of the genes corresponding to feature indices.
    file_path (str): Path of the file where the CSV will be written.
    thres (float): Threshold value for filtering features.

    Returns:
    str: Path of the created CSV file.
    """
    num_spin = features.shape[0]
    # Initialize a variable to record the maximum number of genes observed.
    rec_max_gene = 0
    with open(file_path, 'w', newline='') as file:
        write = csv.writer(file)
        for spin in range(num_spin):
            # Count the number of genes that exceed the threshold for the
            # current spin
            num_gene_show = np.sum(features[spin, :] > thres)
            rec_max_gene = max(rec_max_gene, num_gene_show)
            # Get the indices of the top genes for the current spin
            gene_ind = np.argsort(-features[spin, :])[:num_gene_show]
            cur_line = list(gene_name[gene_ind])
            # Insert the spin number at the beginning of the row
            cur_line.insert(0, spin)
            write.writerow(cur_line)  # Write the current line to the CSV file

    # Read the created CSV, transpose it and save it again
    pd_csv = pd.read_csv(file_path, names=range(rec_max_gene + 1))
    pd_csv = pd_csv.transpose()
    pd_csv.to_csv(file_path, header=False, index=False)

    return file_path


def onmf_gene_program_info(features, gene_name, num_gene_show, fig_folder=None):
    """
    Plot gene program compositions and weights from oNMF.

    Parameters:
    features (numpy.ndarray): oNMF features to be plotted.
    gene_name (List[str]): Names of the genes corresponding to feature indices.
    num_gene_show (int): Number of genes to show in the plot.
    fig_folder (str): Folder where the figure will be saved.

    Returns:
    str: Path of the created figure.
    """
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


def assign_program_position(onmf_rep_ori, umap_all, repulsion=2):
    """
    Assign the position of gene programs on the UMAP plot.

    Args:
    onmf_rep_ori (numpy.ndarray): The gene or gene program representation of the transformed data.
    umap_all (numpy.ndarray): The UMAP coordinates of all cells.
    repulsion (float): The repulsion strength for the layout optimization.

    Returns:
    numpy.ndarray: The assigned positions of gene programs on the UMAP plot.
    """

    num_spin = onmf_rep_ori.shape[1]
    program_umap_pos = np.zeros([num_spin, 2])
    for ii in range(num_spin):
        weight_sub = onmf_rep_ori[:, ii].copy()
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

def gene_program_on_umap(onmf_rep, umap_all, program_umap_pos, fig_folder=None, subsample=True):
    """
    Plot gene programs on the UMAP plot.

    Args:
    onmf_rep (numpy.ndarray): The gene or gene program representation of the transformed data.
    umap_all (numpy.ndarray): The UMAP coordinates of all cells.
    program_umap_pos (numpy.ndarray): The assigned positions of gene programs on the UMAP plot.
    fig_folder (str): The folder where the output figure is saved.
    subsample (bool): Whether to subsample the data for plotting.

    Returns:
    str: Path of the created figure.
    """

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

        plot_data = onmf_rep[:, spin].copy()
        plot_data /= np.percentile(plot_data, 95)
        plot_data = plot_data.clip(0, 3)

        plt.scatter(umap_all[:, 0], umap_all[:, 1], c=plot_data, s=1, 
        alpha=0.5, vmax=1.2, cmap='BuPu', vmin=-0.1)
        plt.text(program_umap_pos[spin, 0], program_umap_pos[spin, 1], str(spin), fontsize=12, path_effects=[patheffects.withStroke(linewidth=3, foreground='w')])
        ax.set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.title(spin)

    if fig_folder is not None:
        plt.savefig(fig_folder + 'gene_program_on_umap.png', dpi=300, bbox_inches='tight')    
        return fig_folder + 'gene_program_on_umap.png'


def visualize_program_expression(onmf_summary, spin_name, gene_matrix, onmf_rep_tri, fig_folder=None, 
                                 num_gene_select=10, n_clusters=5, subsample=10000):
    """ 
    Heatmap comparisons between gene expression matrix and gene program representation.
    
    Args:
    onmf_summary (sklearn.decomposition.NMF): The ONMF summary object.
    spin_name (List[str]): The names of gene programs.
    gene_matrix (numpy.ndarray): The gene expression matrix.
    onmf_rep_tri (numpy.ndarray): The gene program representation.
    fig_folder (str): The folder where the output figure is saved.
    num_gene_select (int): The number of genes to be selected for each gene program. Default is 10.
    n_clusters (int): The number of clusters for KMeans clustering. Default is 5.
    subsample (int): The number cell subsampling for visualization. Default is 10000.

    Returns:
    str: Path of the created figure.
    """
    # Extracting feature components from the ONMF summary object
    features = onmf_summary.components_
    num_spin, num_gene = features.shape

    # Identifying the gene programs by selecting the highest feature component
    # for each gene
    gene_mod_ind = np.argmax(features, axis=0)

    # For each spin, identifying the top selected genes and aggregating them into gene_mod_use
    gene_mod_use = []
    for ind in range(num_spin):
        gene_in_mod = np.where(gene_mod_ind == ind)[0]
        cur_gene = gene_in_mod[np.argsort(- features[ind,
                                          gene_in_mod])[: num_gene_select]]
        gene_mod_use += list(cur_gene)
    gene_mod_use = np.array(gene_mod_use)

    # Creating a subset of cells and reordering them based on KMeans clustering
    np.random.seed(0)
    
    onmf_rep_subset = onmf_rep_tri
    subset_ind = range(gene_matrix.shape[0])
    if subsample:
        if gene_matrix.shape[0] > subsample:
            subset_ind = np.random.choice(range(gene_matrix.shape[0]), size=subsample, replace=False)
            onmf_rep_subset = onmf_rep_tri[subset_ind, :]              

    cell_order = np.argsort(KMeans(n_clusters=n_clusters).fit_predict(onmf_rep_subset))
    gene_matrix_subset = gene_matrix[subset_ind, :][:, gene_mod_use].copy()
    gene_matrix_subset /= np.max(gene_matrix, axis=0)[gene_mod_use].clip(0.2, np.inf)

    sc.set_figure_params(figsize=[10, 5])

    # Plotting Gene Expression Subplot Before Decomposition
    plt.subplot(1, 2, 1)
    plt.imshow(gene_matrix_subset[cell_order, :].T,
               aspect='auto', cmap='Blues', interpolation='none')
    plt.ylabel('Gene')
    plt.xlabel('Cell')
    plt.title('Gene expression')
    plt.grid()

    # Plotting Gene Program Expression Subplot After Decomposition
    plt.subplot(1, 2, 2)
    plt.imshow(onmf_rep_subset[cell_order, :].T,
               aspect='auto', cmap='Blues', interpolation='none')
    plt.yticks(range(num_spin), spin_name, fontsize=12)
    plt.gca().yaxis.set_ticks_position('right')
    plt.xlabel('Cell')
    plt.title('Gene program expression')
    plt.grid()

    if fig_folder is not None:
        plt.savefig(fig_folder + 'gene_program_decomposition.png', bbox_inches='tight')
        return fig_folder + 'gene_program_decomposition.png'

def format_label(label):
    """
    Format the label string by inserting a newline character after every second word.

    Parameters:
    label (str): The label string to be formatted.

    Returns:
    str: The formatted label string.
    """
    parts = label.split('_')
    i = 0
    # Loop through the parts and insert a newline character after every second
    # part
    while i < len(parts) - 1:
        if i % 2 == 1:
            parts[i] = parts[i] + '\n'
        i += 1

    return '_'.join(parts)

def temporary_spin_name(csv_file, num_gene: int = 4):
    """
    Create temporary spin names based on the CSV file content.

    Parameters:
    csv_file (str): Path to the CSV file containing the data.
    num_gene (int): Number of genes to be considered for creating the name.

    Returns:
    List[str]: A list containing the generated temporary spin names.
    """
    df = pd.read_csv(csv_file, header=None)
    # Create spin names by joining the first 4 elements (or less if not
    # available) of each column in the dataframe
    spin_names = ['P' + '_'.join(map(str, df[col][:num_gene])) for col in df.columns]
    spin_names_formatted = [format_label(name) for name in spin_names]
    return spin_names_formatted

def spin_order_in_cluster(j_mat, resolution_parameter: float = 2):
    """ Determine the order of spins in the cluster.

    Args:
    - j_mat: The adjacency matrix representing connections between spins.

    Returns:
    - spin_order: The order of spins determined by the function.
    - pert_pos: The positions perturbed for visualization.
    """
    np.fill_diagonal(j_mat, 0)

    thres = 0
    j_filt = j_mat.copy()
    j_filt[np.abs(j_mat) < thres] = 0
    np.fill_diagonal(j_filt, 0)
    G = nx.from_numpy_array(j_filt)

    G = ig.Graph.from_networkx(G)
    G_pos = G.subgraph_edges(
        G.es.select(
            weight_gt=0),
        delete_vertices=False)
    G_neg = G.subgraph_edges(
        G.es.select(
            weight_lt=0),
        delete_vertices=False)
    G_neg.es['weight'] = [-w for w in G_neg.es['weight']]

    part_pos = la.RBConfigurationVertexPartition(
        G_pos, weights='weight', resolution_parameter=resolution_parameter)
    part_neg = la.RBConfigurationVertexPartition(
        G_neg, weights='weight', resolution_parameter=resolution_parameter)
    optimiser = la.Optimiser()
    diff = optimiser.optimise_partition_multiplex(
        [part_pos, part_neg], layer_weights=[1, -1])

    net_class = list(part_pos)
    spin_order = [spin for cur_list in net_class for spin in cur_list]
    net_class_len = [len(cur_list) for cur_list in net_class]

    start_angle = 0 * np.pi
    end_angle = 2 * np.pi
    gap_size = 2

    angle_list_raw = np.linspace(start_angle, end_angle, np.sum(
        net_class_len) + gap_size * len(net_class_len) + 1)[: - 1]
    angle_list = []
    size_group_cum = np.cumsum(net_class_len)
    size_group_cum = np.insert(size_group_cum, 0, 0)
    # angle_list = np.linspace(start_angle, end_angle, len(leiden_list) + 1)
    for ii in range(len(net_class_len)):
        angle_list.extend(
            angle_list_raw[size_group_cum[ii] + gap_size * ii: size_group_cum[ii + 1] + gap_size * ii])

    pert_dist = 3

    pert_pos = np.array(
        [- pert_dist * np.cos(angle_list), pert_dist * np.sin(angle_list)]).T

    return spin_order, pert_pos

def plot_network(
        G,
        j_mat,
        ax,
        pos: None,
        nodesz=1,
        linewz=1,
        node_color='k'):
    """ 
    Plot the network.

    Args:
    - G: The networkx graph object.
    - j_mat: The adjacency matrix representing connections between spins.
    - ax: The axis object to plot the network.
    - nodesz: The size of nodes in the network.
    - linewz: The width of edges in the network.
    - node_color: The color of nodes in the network.
    - pos: The positions of nodes in the network.
    """

    self_loops = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(self_loops)

    eposi = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
    wposi = np.array([d['weight']
                        for (u, v, d) in G.edges(data=True) if d['weight'] > 0])

    enega = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0]
    wnega = np.array([d['weight']
                        for (u, v, d) in G.edges(data=True) if d['weight'] < 0])

    col1 = '#f0dab1'

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=61.8 *
        nodesz,
        node_color=node_color,
        edgecolors='k')

    def sig_fun(xx): return (1 / (1 + np.exp(- 5 * (xx + cc))))
    cc = np.max(np.abs(j_mat)) / 10

    # edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=eposi,
        width=linewz * wposi,
        edge_color='#3285CC',
        alpha=sig_fun(wposi))

    nx.draw_networkx_edges(G,
                            pos,
                            ax=ax,
                            edgelist=enega,
                            width=- linewz * wnega,
                            edge_color='#E84B23',
                            alpha=sig_fun(- wnega))

    margin = 0.2
    plt.margins(x=0.1, y=0.1)

    ax.set_axis_off()
    ax.set_aspect('equal')
    return ax

def adjust_label_position(pos, offset=0.1):
    """ 
    Adjust the label positions radially outward from the center.

    Args:
    - pos: The original positions of the labels.
    - offset: The radial distance by which to adjust the label positions.

    Returns:
    - adjusted_pos: The adjusted positions of the labels.
    """
    adjusted_pos = {}
    for node, coordinates in enumerate(pos):
        theta = np.arctan2(coordinates[1], coordinates[0])
        radius = np.sqrt(coordinates[0]**2 + coordinates[1]**2)
        adjusted_pos[node] = (
            coordinates[0] + np.cos(theta) * offset,
            coordinates[1] + np.sin(theta) * offset)
    return adjusted_pos

def plot_final(
        cur_j,
        gene_program_name,
        cluster:bool = True,
        title: str = "Gene Regulatory Network Reconstructed by D-SPIN",
        adj_matrix_threshold: float = 0.4,
        resolution_parameter: float = 2,
        nodesz: float = 3,
        linewz: float = 2,
        node_color: str = 'k',
        figsize=[20, 20],
        pos=None,
        spin_order=None,
        node_fontsize=None):
    """
    Plot the final gene regulatory network.

    Args:
    - gene_program_name: The names of gene programs.
    - cur_j: The adjacency matrix representing connections between gene programs.
    - nodesz: The size of nodes in the network.
    - linewz: The width of edges in the network.
    - node_color: The color of nodes in the network.
    - pos: The positions of nodes in the network.
    """
    sc.set_figure_params(figsize=figsize)

    num_spin = cur_j.shape[0]
    
    # Setting default values for parameters of node size and line width.
    if not nodesz:
        nodesz = np.sqrt(100 / num_spin)
    if not linewz:
        linewz = np.sqrt(100 / num_spin)
    if node_fontsize is None:
        node_fontsize = figsize[0] * 20/ num_spin


    fig, grid = sc.pl._tools._panel_grid(0.2, 0.2, ncols=2, num_panels=2)

    # Filtering the adjacency matrix.
    cur_j_filt = cur_j.copy()
    cur_j_filt[np.abs(cur_j_filt) < np.percentile(np.abs(cur_j_filt), adj_matrix_threshold * 100)] = 0

    if spin_order is None:
        # Calculating spin orders and perturbed positions for plotting.
        spin_order, pert_pos = spin_order_in_cluster(cur_j, resolution_parameter)

    # Creating a graph from the filtered adjacency matrix and ordering spins.
    G = nx.convert_matrix.from_numpy_array(cur_j_filt[spin_order, :][:, spin_order])

    # Initializations and adjustments for plotting.
    if not cluster:
        pos = nx.circular_layout(G)
    else:
        pos = pert_pos
    node_color = ['#f0dab1'] * num_spin
    node_label = np.array(gene_program_name)[spin_order]
    # node_label = np.array([format_label(label) for label in gene_list])


    ax = plt.subplot(grid[1])
    ax = plot_network(
        G,
        cur_j_filt,
        ax,
        nodesz=nodesz,
        linewz=linewz,
        node_color=node_color,
        pos=pos)

    # Adding labels with path effects to nodes in the network.
    path_effect = [patheffects.withStroke(linewidth=3, foreground='w')]
    
    if cluster:
        adjusted_positions = adjust_label_position(pos, 0.5)
    else:
        adjusted_positions = pos
    for ii in range(num_spin):
        x, y = adjusted_positions[ii]
        text = plt.text(
            x,
            y,
            node_label[ii],
            fontsize= node_fontsize,
            color='k',
            ha='center',
            va='center',
            rotation=np.arctan(
                pos[ii][1] /
                pos[ii][0]) /
            np.pi *
            180)
        text.set_path_effects(path_effect)
    if not title:
        title = 'Gene Regulatory Network Reconstructed by D-SPIN'
    ax.set_title(title, fontsize=node_fontsize * 1.5, y = 1.05)


