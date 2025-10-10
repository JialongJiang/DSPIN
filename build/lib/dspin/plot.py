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
import csv
import scanpy as sc
import igraph as ig
import leidenalg as la
import networkx as nx
from scipy import optimize
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib as mpl
from typing import List, Dict, Optional, Tuple


def onmf_to_csv(features, gene_name, file_path, thres=0.01, if_write_weights=False):
    """
    Extract gene names in ONMF to CSV.

    Parameters:
    features (numpy.ndarray): oNMF features to be written to CSV.
    gene_name (List[str]): Names of the genes corresponding to feature indices.
    file_path (str): Path of the file where the CSV will be written.
    thres (float): Threshold value for filtering features.
    if_write_weights (bool): If True, writes weights to the CSV file.

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

            if if_write_weights:
                cur_line = [features[spin, ind].round(6) for ind in gene_ind]
                cur_line.insert(0, f'Weights_{spin}')
                write.writerow(cur_line)

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


def plot_network_heatmap(j_mat: np.ndarray, 
                       module_list: List[List[int]] = None, 
                       spin_name_list: Optional[List[str]] = None,
                       fig_folder: Optional[str] = None) -> None:
    """
    Plot a heatmap of the network adjacency matrix with modules highlighted.
    
    Parameters
    ----------
    j_mat : np.ndarray
        The network adjacency matrix.
    modules : List[List[int]]
        List of modules, where each module is a list of node indices.
    spin_name_list : List[str], optional
        List of names for the spins/nodes. If None, will use generic names.
    fig_folder : str, optional
        Folder path to save the figure. If None, will not save the figure.
    """
    num_spin = j_mat.shape[0]
    j_mat = j_mat.copy()
    np.fill_diagonal(j_mat, 0)
    
    if module_list is None:
        module_list = [list(np.arange(num_spin))]

    # Create spin order based on modules
    spin_order = [spin for cur_list in module_list for spin in cur_list]
    net_class_len = [len(cur_list) for cur_list in module_list]
    
    # Create custom colormap
    cmap_hvec = mpl.colors.LinearSegmentedColormap.from_list("", ['#E84B23', '#FFFFFF', '#3285CC'])
    
    color_thres = np.percentile(np.abs(j_mat[j_mat != 0]), 95)

    # Plot heatmap
    sc.set_figure_params(figsize=[0.18 * num_spin + 0.5, 0.18 * num_spin])
    plt.imshow(j_mat[:, spin_order][spin_order, :], cmap=cmap_hvec, vmin=- color_thres, vmax=color_thres)
    
    # Add module boundaries
    for ii in range(len(net_class_len)):
        plt.axhline(np.sum(net_class_len[:ii]) - 0.5, color='k', linewidth=1)
        plt.axvline(np.sum(net_class_len[:ii]) - 0.5, color='k', linewidth=1)
    
    fsize = 9

    # Set labels
    if spin_name_list is not None:
        plt.xticks(np.arange(len(spin_order)), np.array(spin_name_list)[spin_order], rotation=90, fontsize=fsize)
        plt.yticks(np.arange(len(spin_order)), np.array(spin_name_list)[spin_order], fontsize=fsize)
    else:
        plt.xticks(np.arange(len(spin_order)), [f'P{spin}' for spin in spin_order], rotation=90, fontsize=fsize)
        plt.yticks(np.arange(len(spin_order)), [f'P{spin}' for spin in spin_order], fontsize=fsize)
        
    plt.colorbar(fraction=0.6 / num_spin)
    plt.grid()
    
    # Save figure if folder is provided
    if fig_folder is not None:
        plt.savefig(fig_folder + 'network_heatmap.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_response_heatmap(h_vec: np.ndarray, 
                       module_list: List[List[int]]=None, 
                       spin_name_list: Optional[List[str]] = None,
                       sample_list: Optional[List[str]] = None,
                       fig_folder: Optional[str] = None) -> None:
    """
    Plot a heatmap of the response matrix with modules highlighted.
    
    Parameters
    ----------
    h_vec : np.ndarray
        The response vector.
    modules : List[List[int]]
        List of modules, where each module is a list of node indices.
    spin_name_list : List[str], optional
        List of names for the spins/nodes. If None, will use generic names.
    sample_list : List[str], optional
        List of names for the samples. If None, will use generic names.
    fig_folder : str, optional
        Folder path to save the figure. If None, will not save the figure.
    """
    num_spin, num_sample = h_vec.shape
    h_vec = h_vec.copy()
    
    if module_list is None:
        module_list = [list(leaves_list(linkage(h_vec, method='ward')))]

    # Create spin order based on modules
    spin_order = [spin for cur_list in module_list for spin in cur_list]
    net_class_len = [len(cur_list) for cur_list in module_list]
    sample_order = leaves_list(linkage(h_vec.T, method='ward'))
    
    cmap_hvec = mpl.colors.LinearSegmentedColormap.from_list("", ['#E84B23', '#FFFFFF', '#3285CC'])
    
    # Plot heatmap
    sc.set_figure_params(figsize=[0.16 * num_spin + 0.5, 0.16 * num_sample])
    plt.imshow(h_vec.T[sample_order, :][:, spin_order], cmap=cmap_hvec, vmin=- 1, vmax=1, aspect='auto')
    
    # Add module boundaries
    for ii in range(len(net_class_len)):
        plt.axvline(np.sum(net_class_len[:ii]) - 0.5, color='k', linewidth=1)
    
    # Set labels
    if spin_name_list is not None:
        plt.xticks(np.arange(len(spin_order)), np.array(spin_name_list)[spin_order], rotation=90, fontsize=8)
    else:
        plt.xticks(np.arange(len(spin_order)), [f'P{spin}' for spin in spin_order], rotation=90, fontsize=8)
    
    plt.yticks(np.arange(len(sample_order)), np.array(sample_list)[sample_order], fontsize=8)
        
    plt.colorbar(fraction=1 / num_spin)
    plt.grid()
    
    # Save figure if folder is provided
    if fig_folder is not None:
        plt.savefig(fig_folder + 'module_heatmap.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def create_undirected_network(j_mat, node_names, thres_strength=0.05):

    np.fill_diagonal(j_mat, 0)
        
    # Filter weak connections
    j_filt = j_mat.copy()
    j_filt[np.abs(j_mat) < thres_strength] = 0
    np.fill_diagonal(j_filt, 0)
    
    # Create networkx graph
    G = nx.from_numpy_array(j_filt)
    G = nx.relabel_nodes(G, {ii: node_names[ii] for ii in range(len(node_names))})

    return G, j_filt


def create_directed_network(j_mat, node_names,thres_strength=0.05, thres_direction=0.05):

    np.fill_diagonal(j_mat, 0)

    num_spin = j_mat.shape[0]
    j_mat_directed = np.zeros(j_mat.shape)

    for ind1 in range(num_spin):
        for ind2 in range(num_spin):
            if np.abs(j_mat[ind1, ind2]) > (1 + thres_direction) * np.abs(j_mat[ind2, ind1]):
                j_mat_directed[ind1, ind2] = j_mat[ind1, ind2]
            elif np.abs(j_mat[ind2, ind1]) > (1 + thres_direction) * np.abs(j_mat[ind1, ind2]):
                j_mat_directed[ind2, ind1] = j_mat[ind2, ind1]
            else:
                j_mat_directed[ind1, ind2] = (j_mat[ind1, ind2] + j_mat[ind2, ind1]) / 2
                j_mat_directed[ind2, ind1] = j_mat_directed[ind1, ind2]

    j_mat_directed[np.abs(j_mat_directed) < thres_strength] = 0
    
    G = nx.from_numpy_array(j_mat_directed.T, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {ii: node_names[ii] for ii in range(len(node_names))})

    return G, j_mat_directed


def layout_loss_fun(xx, ori_pos, network):

    stength_module = 12
    strength_repulsion = 12
    strength_spring = 1

    spring_length = 2
    spring_weight = np.abs(network)

    module_length = 1

    xx = xx.reshape(- 1, 2)
    module_dist = np.clip(np.sqrt(((xx - ori_pos) ** 2).sum(axis=1)) - module_length, 0, None)
    module_attract = np.sum(module_dist ** 2)
    repulse = np.sum(1 / pdist(xx))

    pairwise_distance = squareform(pdist(xx))
    clip_dist = np.clip(pairwise_distance - spring_length, 0, None)
    spring_energy = np.sum(clip_dist ** 2 * spring_weight)

    return stength_module * module_attract + strength_repulsion * repulse + strength_spring * spring_energy


def remove_peripheral_nodes(G, min_in_degree=1, min_out_degree=1, max_iter=100):

    cur_size = len(G.nodes)

    for repeat in range(max_iter):

        remove_nodes1 = [node for node, out_degree in G.out_degree() if out_degree <= min_out_degree]
        remove_nodes2 = [node for node, in_degree in G.in_degree() if in_degree <= min_in_degree]
        remove_nodes = list(set(remove_nodes1 + remove_nodes2))
        G.remove_nodes_from(remove_nodes)

        if cur_size == len(G.nodes):
            break
        else:
            cur_size = len(G.nodes)

    print(f'{len(G.nodes)} nodes after filtering ')

    return G


def compute_modules(G_nx, resolution: float = 1.0,
                        seed: int = 1) -> List[List[int]]:
    """
    Compute network modules using Leiden community detection.
    
    Parameters
    ----------
    resolution : float, optional
        Resolution parameter for Leiden community detection. Higher values lead to more modules. Default is 1.0.
    seed : int, optional
        Random seed for Leiden community detection. Default is 0.
    thres : float, optional
        Threshold for filtering weak connections. Default is 0.
        
    Returns
    -------
    List[List[int]]
        List of modules, where each module is a list of node indices.
    """

    adj_matrix = nx.to_numpy_array(G_nx, weight='weight')
    sub_gene_list = list(G_nx.nodes)

    G = nx.from_numpy_array(adj_matrix)

    G = ig.Graph.from_networkx(G)
    
    # Separate positive and negative edges
    G_pos = G.subgraph_edges(G.es.select(weight_gt=0), delete_vertices=False)
    G_neg = G.subgraph_edges(G.es.select(weight_lt=0), delete_vertices=False)
    G_neg.es['weight'] = [-w for w in G_neg.es['weight']]
    
    # Apply Leiden community detection
    part_pos = la.RBConfigurationVertexPartition(G_pos, weights='weight', resolution_parameter=resolution)
    part_neg = la.RBConfigurationVertexPartition(G_neg, weights='weight', resolution_parameter=resolution)
    
    optimiser = la.Optimiser()
    optimiser.set_rng_seed(seed)
    
    diff = optimiser.optimise_partition_multiplex([part_pos, part_neg], layer_weights=[1, -1])
    
    # Extract modules
    net_class = list(part_pos)
    
    print(f'Identified {len(net_class)} modules with sizes {np.array([len(cur_list) for cur_list in net_class])}')

    abs_degree = np.abs(adj_matrix).sum(axis=1)

    # rank node inside each module by degree
    for ii in range(len(net_class)):
        cur_list = net_class[ii]
        cur_list = sorted(cur_list, key=lambda xx: abs_degree[xx], reverse=True)
        net_class[ii] = cur_list

    return net_class


def compute_module_angle(module_size, gap_size):

    module_plots_size = np.sqrt(module_size)

    angle_list = []
    cur_angle = 0

    for ii in range(len(module_plots_size)):
        angle_list.append(cur_angle)
        cur_angle += module_plots_size[ii] + gap_size

    angle_list = np.array(angle_list)
    angle_list /= (cur_angle / (2 * np.pi))

    return angle_list


def compute_network_positions(network, net_class, circle_size=None, module_angle=None, gap_size=0.5):

    module_size = np.array([len(cur_list) for cur_list in net_class])
    num_spin = np.sum(module_size)

    if module_angle is None:
        module_angle = compute_module_angle(module_size, gap_size)

    if circle_size is None:
        circle_size = num_spin ** 0.3

    module_center = circle_size * np.array([np.cos(module_angle), np.sin(module_angle)]).T

    pos_module = np.zeros((num_spin, 2))

    for ii in range(len(module_size)):
        cur_list = net_class[ii]
        pos_module[cur_list, :] = module_center[ii, :]

    pos_module += np.random.normal(0, 0.05, pos_module.shape)

    ori_pos = pos_module.copy()

    opt_res = optimize.minimize(lambda xx : layout_loss_fun(xx, ori_pos, network), pos_module.flatten())
    pos_raw = opt_res.x.reshape(- 1, 2)

    pos = {ii: pos_raw[ii] for ii in range(num_spin)}

    return pos


def plot_network_diagram(j_mat: np.ndarray,
                        modules: List[List[int]],
                        directed: bool = False,
                        pos: Optional[Dict[int, Tuple[float, float]]] = None, 
                        weight_thres: float = None,
                        spin_name_list_short: Optional[List[str]] = None,
                        fig_folder: Optional[str] = None) -> None:
    """
    Plot the network diagram with modules highlighted.
    
    Parameters
    ----------
    j_mat : np.ndarray
        The network adjacency matrix.
    modules : List[List[int]]
        List of modules, where each module is a list of node indices.
    directed : bool, optional
        Whether the network is directed. If True, will plot the network as a directed network.
    pos : Dict[int, Tuple[float, float]], optional
        Dictionary mapping node indices to (x, y) positions. If None, positions will be computed automatically.
    weight_thres : float, optional
        Threshold for filtering weak connections. If None, will filter by the 10th percentile of the absolute nonzero values of the network adjacency matrix.
    spin_name_list_short : List[str], optional
        Short names for the spins/nodes (used for labels). If None, will use generic names.
    fig_folder : str, optional
        Folder path to save the figure. If None, will not save the figure.
    """
    num_spin = j_mat.shape[0]
    
    # Filter network for visualization
    if weight_thres is None:
        thres = np.percentile(np.abs(j_mat[j_mat != 0]), 10)
    else:
        thres = weight_thres

    j_filt = j_mat.copy()
    j_filt[np.abs(j_mat) < thres] = 0
    np.fill_diagonal(j_filt, 0)
    
    if directed:
        G = nx.from_numpy_array(j_filt, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(j_filt)
    
    # Compute positions
    if pos is None:
        pos = compute_network_positions(j_mat, modules)
    
    # Create labels
    if spin_name_list_short is not None:
        label = {ii: spin_name_list_short[ii] for ii in range(num_spin)}
    else:
        label = {ii: f'P{ii}' for ii in range(num_spin)}
    
    # Plot network
    fig_size = 4 * num_spin ** 0.25
    sc.set_figure_params(figsize=[fig_size, fig_size])
    fig, ax = plt.subplots()
    
    # Draw edges
    eposi = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
    wposi = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if d['weight'] > 0])
    
    enega = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0]
    wnega = np.array([d['weight'] for (u, v, d) in G.edges(data=True) if d['weight'] < 0])
    
    linewz = 0.5 / thres 

    # Draw positive edges
    if len(eposi) > 0:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=eposi, width=linewz * wposi,
                               edge_color='#3285CC', alpha=0.7)
    
    # Draw negative edges
    if len(enega) > 0:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=enega, width=linewz * (-wnega),
                               edge_color='#E84B23', alpha=0.7)
    
    # Draw nodes
    node_colors = ['#f0dab1'] * num_spin
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color=node_colors, 
                          edgecolors='black', linewidths=1)
    
    # Draw labels
    for spin in range(num_spin):
        plt.text(pos[spin][0], pos[spin][1], label[spin], fontsize=10, 
                path_effects=[patheffects.withStroke(linewidth=3, foreground='w')], 
                ha='center', va='center')
    
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Save figure if folder is provided
    if fig_folder is not None:
        plt.savefig(fig_folder + 'network_diagram.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_top_regulators(program_interactions: np.ndarray, 
                        program_index: int, 
                        gene_list: np.array, 
                        program_name: Optional[str] = None, 
                        thres: Optional[float] = None, 
                        fig_folder: Optional[str] = None) -> None:
    
    """
    Plot the top regulators of a program.

    Parameters
    ----------
    program_interactions : np.ndarray
        The program interactions.
    program_index : int
        The index of the program.
    gene_list : np.array
        The list of genes.
    program_name : Optional[str], optional
        The name of the program.
    thres : Optional[float], optional
        The threshold for the regulators.
    fig_folder : Optional[str], optional
        The folder to save the figure.
    """

    cur_regulators = program_interactions[:, program_index]
    if thres is None:
        thres = np.sort(np.abs(cur_regulators))[::-1][20]

    num_gene_show_pos = np.sum(cur_regulators >= thres)
    num_gene_show_neg = np.sum(cur_regulators <= -thres)
    num_gene_show = num_gene_show_pos + num_gene_show_neg

    plot_ind_pos = np.argsort(cur_regulators)[::-1][:num_gene_show_pos]
    plot_ind_neg = np.argsort(cur_regulators)[:num_gene_show_neg]

    plot_ind = np.concatenate([plot_ind_pos, plot_ind_neg])

    sc.set_figure_params(figsize=(num_gene_show * 0.17, 2))

    plt.scatter(np.arange(num_gene_show), cur_regulators[plot_ind], c=np.where(cur_regulators[plot_ind] > 0,'#3285CC', '#E84B23'), zorder=10)
    plt.xticks(np.arange(num_gene_show), np.array(gene_list)[plot_ind], rotation=90, fontsize=9); 
    plt.yticks(fontsize=10)
    plt.ylabel('Regulator strength', fontsize=10)
    plt.xlabel('Regulator', fontsize=10)

    if program_name is not None:
        plt.title(program_name, fontsize=10)

    if fig_folder is not None:
        plt.savefig(fig_folder + 'top_regulators.png')

    plt.show()    