# -*-coding:utf-8 -*-
'''
@Time    :   2023/10/13 10:43
@Author  :   Jialong Jiang, Yingying Gong
'''

from .plot import onmf_to_csv
from .compute import (
    onmf_discretize,
    sample_corr_mean,
    sample_states,
    learn_network_adam,
    learn_program_regulators,
    compute_onmf_decomposition,
    subsample_normalize_gene_matrix,
    summary_components,
    compute_relative_responses
)
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from tqdm import tqdm
from scipy.io import savemat, loadmat
from scipy.sparse import issparse
import networkx as nx
import matplotlib.patheffects as patheffects
import warnings
import itertools
from typing import List, Dict, Optional, Tuple, Union


class AbstractDSPIN(ABC):
    """
    Abstract base class for DSPIN, providing common methods and properties for GeneDSPIN and ProgramDSPIN.
    
    Attributes
    ----------
    adata : ad.AnnData
        Annotated single-cell gene expression data.
    save_path : str
        Directory path where results will be stored.
    num_spin : int
        Number of spins for modeling.
    """

    def __init__(self,
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int):
        """
        Initialize an instance of AbstractDSPIN.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data matrix containing observations and gene expressions.
        save_path : str
            Directory path for saving results.
        num_spin : int
            Number of spins used in DSPIN.
        """

        self.adata = adata
        self.save_path = os.path.abspath(save_path) + '/'
        self.num_spin = num_spin

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("Saving path does not exist. Creating a new folder.")
            
        self.fig_folder = self.save_path + 'figures/'
        os.makedirs(self.fig_folder, exist_ok=True)

    @property
    def program_representation_raw(self):
        return self._onmf_rep_ori

    @property
    def program_representation(self):
        return self._onmf_rep_tri

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def network(self):
        return self._network

    @property
    def responses(self):
        return self._responses
    
    @property
    def relative_responses(self):
        return self._relative_responses

    @property
    def sample_list(self):
        return self._samp_list

    @program_representation_raw.setter
    def program_representation_raw(self, value):
        self._onmf_rep_ori = value

    @program_representation.setter
    def program_representation(self, value):
        self._onmf_rep_tri = value

    @network.setter
    def network(self, value):
        self._network = value

    @responses.setter
    def responses(self, value):
        self._responses = value

    @relative_responses.setter
    def relative_responses(self, value):
        self._relative_responses = value

    @sample_list.setter
    def sample_list(self, value):
        self._samp_list = value

    def discretize(self, clip_percentile: float = 100) -> None:
        """
        Discretizes the oNMF representation into three states (-1, 0, 1) using K-means clustering.

        Parameters
        ----------
        clip_percentile : float, optional
            Percentile at which to clip values before discretization (default is 100).
        """
        onmf_rep_ori = self.program_representation_raw.copy()
        fig_folder = self.fig_folder

        if clip_percentile < 100:
            for ii in range(onmf_rep_ori.shape[1]):
                onmf_rep_ori[:, ii] = onmf_rep_ori[:, ii] / np.percentile(onmf_rep_ori[:, ii], clip_percentile)
            onmf_rep_ori = onmf_rep_ori.clip(0, 1)

        onmf_rep_tri = onmf_discretize(onmf_rep_ori, fig_folder)
        self._onmf_rep_tri = onmf_rep_tri

    def raw_data_corr(self, sample_id_key: str) -> None:
        """
        Computes correlation of raw data based on a sample column name.

        Parameters
        ----------
        sample_id_key : str
            Column name in adata.obs specifying sample identifiers.
        """
        raw_data, samp_list = sample_corr_mean(
            self.adata.obs[sample_id_key], self._onmf_rep_tri)
        self._raw_data = raw_data
        self._samp_list = samp_list

    def raw_data_state(self, sample_id_key) -> None:
        """
        Compute and return the state of raw data based on sample identifiers.

        Parameters
        ----------
        sample_id_key : str
            Column name in `adata.obs` used to group and compute raw data states.
        """
        state_list, samp_list = sample_states(
            self.adata.obs[sample_id_key], self._onmf_rep_tri)
        
        self._raw_data = state_list
        self._samp_list = samp_list

    def default_params(self,
                       method: str) -> dict:
        """
        Provide the default parameters for the specified algorithm.

        Parameters
        ----------
        method : str
            The inference method. Options: 'maximum_likelihood', 'mcmc_maximum_likelihood', 'pseudo_likelihood'.

        Returns:
        dict: Dictionary containing default parameters for the specified method.
        """

        num_spin = self.num_spin
        raw_data = self._raw_data

        if self.example_list is not None:
            example_list_ind = [list(self._samp_list).index(samp)
                                for samp in self.example_list]
            self._samp_list = self._samp_list[example_list_ind]
            raw_data = raw_data[example_list_ind]
            self._raw_data = raw_data

        num_sample = len(raw_data)
        params = {'num_epoch': 400,
                  'cur_j': np.zeros((num_spin, num_spin)),
                  'cur_h': np.zeros((num_spin, num_sample)),
                  'save_path': self.save_path, 
                  'rec_gap': 10, 
                  'seed': 0}
        params.update({'lambda_l1_j': 0.01,
                       'lambda_l1_h': 0,
                       'lambda_l2_j': 0,
                       'lambda_l2_h': 0.005})
        params.update({'backtrack_gap': 40,
                       'backtrack_tol': 5})

        if method == 'maximum_likelihood':
            params['stepsz'] = 0.2
        elif method == 'mcmc_maximum_likelihood':
            params['stepsz'] = 0.02
            params['mcmc_samplingsz'] = 2e5
            params['mcmc_samplingmix'] = 1e3
            params['mcmc_samplegap'] = 1
        elif method == 'pseudo_likelihood':
            params['stepsz'] = 0.05

        return params

    def network_inference(self,
                          sample_id_key: str = 'sample_id',
                          method: str = 'auto',
                          directed: bool = False,
                          params: dict = None,
                          example_list: List[str] = None,
                          run_with_matlab: bool = False) -> None:
        """
        Perform network inference using a specified method and parameters.

        Parameters
        ----------
        sample_id_key : str, optional
            Column name in `adata.obs` for sample identifiers. Default is 'sample_id'.
        method : str, optional
            The method used for network inference. Options: 'maximum_likelihood', 'mcmc_maximum_likelihood', 'pseudo_likelihood', 'auto'. Default is 'auto'.
        directed : bool, optional
            Whether to infer a directed network. Default is False.
        params : dict, optional
            Additional parameters for network inference. Default is None.
        example_list : List[str], optional
            List of example sample identifiers. Default is None.
        run_with_matlab : bool, optional
            If True, prepares data for MATLAB execution instead of running inference in Python. Default is False.
        """

        self.sample_id_key = sample_id_key

        if method not in [
            'maximum_likelihood',
            'mcmc_maximum_likelihood',
            'pseudo_likelihood',
            'auto']:
            raise ValueError(
                "Method must be one of 'maximum_likelihood', 'mcmc_maximum_likelihood', 'pseudo_likelihood' or 'auto'.")

        if method == 'auto':           
            samp_list = np.unique(self.adata.obs[sample_id_key])
            num_sample = len(samp_list)
            if example_list is not None:
                num_sample = len(example_list)
            if num_sample > 30:
                method = 'pseudo_likelihood'
            else:
                if self.num_spin <= 12:
                    method = 'maximum_likelihood'
                elif self.num_spin <= 25:
                    method = 'mcmc_maximum_likelihood'
                else:
                    method = 'pseudo_likelihood'
            if directed:
                method = 'pseudo_likelihood'

        print("Using {} for network inference.".format(method))

        if method == 'pseudo_likelihood':
            self.raw_data_state(sample_id_key)
        else:
            self.raw_data_corr(sample_id_key)

        self.example_list = example_list

        train_dat = self.default_params(method)
        train_dat['directed'] = directed
        if params is not None:
            train_dat.update(params)

        if run_with_matlab:

            file_path = self.save_path + 'raw_data.mat'
            savemat(file_path, {'raw_data': self._raw_data, 'sample_list': self._samp_list, 
            'method': method, 'directed': directed, **train_dat})
            print("Data saved to {}. Please run the network inference in MATLAB".format(file_path))

        else:
            cur_j, cur_h, train_log = learn_network_adam(self.raw_data, method, train_dat)
            self._network = cur_j
            self._responses = cur_h
            self.train_log = train_log

    def response_relative_to_control(self, 
                           sample_id_key: str = 'sample_id',
                           if_control_key: str = 'if_control', 
                           batch_key: str = 'batch'):
        """
        Compute the relative responses based on control samples in each batch.

        Parameters
        ----------
        sample_id_key : str, optional
            Column name in `adata.obs` specifying sample identifiers. Default is 'sample_id'.
        if_control_key : str, optional
            Column name in `adata.obs` indicating whether a sample is a control. Default is 'if_control'.
        batch_key : str, optional
            Column name in `adata.obs` specifying batch assignment for each sample. Default is 'batch'.
        """

        if if_control_key not in self.adata.obs.columns:
            raise ValueError(f"Column '{if_control_key}' not found in adata.obs.")
        if batch_key not in self.adata.obs.columns:
            raise ValueError(f"Column '{batch_key}' not found in adata.obs.")
        
        sample_list = self._samp_list
        sample_to_control = self.adata.obs.groupby(sample_id_key)[if_control_key].first().to_dict()
        sample_to_batch = self.adata.obs.groupby(sample_id_key)[batch_key].first().to_dict()

        if_control = np.array([sample_to_control[sample] for sample in sample_list])
        batch_index = np.array([sample_to_batch[sample] for sample in sample_list])

        self._relative_responses = compute_relative_responses(self.responses, if_control, batch_index)


class GeneDSPIN(AbstractDSPIN):
    """
    GeneDSPIN inherits the AbstractDSPIN class, optimized for building gene-level networks.
    It doesn't need oNMF and can directly perform network inference on the genes.
    """

    def __init__(self,
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int,
                 clip_percentile: float = 95):
        """
        Initialize the GeneDSPIN object.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated single-cell gene expression data.
        save_path : str
            Directory path where results will be saved.
        num_spin : int
            Number of spins used in inference.
        clip_percentile : float, optional
            Percentile threshold for clipping values before discretization. Default is 95.
        """

        super().__init__(adata, save_path, num_spin)
        print("GeneDSPIN initialized.")

        if issparse(adata.X):
            self._onmf_rep_ori = adata.X.toarray()
        else:
            self._onmf_rep_ori = adata.X

        self.discretize(clip_percentile)

    def program_regulator_discovery(self, 
                                    program_representation_raw: np.ndarray, 
                                    sample_id_key: str = 'sample_id',
                                    params: dict = None):
        """
        Discover regulators of given gene programs using regression.

        Parameters
        ----------
        program_representation_raw : np.ndarray
            The gene program representation.
        sample_id_key : str, optional
            Column name in `adata.obs` specifying sample identifiers. Default is 'sample_id'.
        params : dict, optional
            Additional parameters for regression. Default is None.
        """

        program_states, samp_list = sample_states(self.adata.obs[sample_id_key], program_representation_raw)

        gene_states, _ = sample_states(self.adata.obs[sample_id_key], self._onmf_rep_tri)

        train_dat = {
            'num_epoch': 500,
            'stepsz': 0.01,
            'lambda_l1_interaction': 0.01,
            'rec_gap': 10
        }

        if params is not None:
            train_dat.update(params)

        cur_interaction, cur_selfj, cur_selfh = learn_program_regulators(gene_states, program_states, train_dat)

        self.program_interactions = cur_interaction
        self.program_activities = cur_selfh
        self.program_self_interactions = cur_selfj
        


class ProgramDSPIN(AbstractDSPIN):
    """
    ProgramDSPIN extends AbstractDSPIN to handle large datasets efficiently.
    This class includes methods for subsampling, balancing gene matrices, and running repeated oNMF decompositions.

    Attributes
    ----------
    adata : ad.AnnData
        Annotated single-cell gene expression data.
    save_path : str
        Directory path where results will be stored.
    num_spin : int
        Number of spins used in modeling.
    num_onmf_components : int, optional
        Number of oNMF components.
    prior_programs : List[List[str]], optional
        List of predefined gene programs.
    num_repeat : int, optional
        Number of oNMF repetitions for stability.
    """

    def __init__(self,
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int,
                 num_onmf_components: int = None,
                 prior_programs: List[List[str]] = None,
                 num_repeat: int = 10):
        """
        Initialize the ProgramDSPIN object.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data matrix containing observations and gene expressions.
        save_path : str
            Directory path for saving results.
        num_spin : int
            Number of spins used in DSPIN.
        num_onmf_components : int, optional
            Number of oNMF components, default is None.
        prior_programs : List[List[str]], optional
            List of predefined gene programs.
        num_repeat : int, optional
            Number of repetitions for oNMF computation. Default is 10.
        """
        super().__init__(adata, save_path, num_spin)

        print("ProgramDSPIN initialized.")

        self._onmf_summary = None
        self._gene_matrix_large = None
        self._use_data_list = None
        self.gene_program_csv = None
        self.prior_programs = None
        self.preprogram_num = len(prior_programs) if prior_programs else 0

    @property
    def onmf_decomposition(self):
        return self._onmf_summary

    @onmf_decomposition.setter
    def onmf_decomposition(self, value):
        self._onmf_summary = value

    def summarize_onmf_result(self,
                              large_subsample_matrix: np.ndarray,
                              num_onmf_components: int,
                              num_repeat: int) -> NMF:
        """
        Summarize the results of repeated oNMF decompositions,
        integrating components from each oNMF run and prior programs if available.

        Parameters
        ----------
        num_onmf_components : int
            Number of oNMF components.
        num_repeat : int
            Number of oNMF repetitions.

        Returns
        -------
        NMF
            An NMF object containing the summarized components.
        """

        # Retrieve the number of spins and annotated data
        num_spin = self.num_spin
        adata = self.adata
        prior_programs_mask = self.prior_programs_mask
        seed_list = self.onmf_parameters['seed_list']
        summary_method = self.onmf_parameters['summary_method']

        rec_components = np.zeros((num_repeat, num_onmf_components, np.sum(prior_programs_mask)))

        for ii in range(num_repeat):

            cur_seed = seed_list[ii]
            cur_onmf = np.load(self.save_path + f'onmf/onmf_components_{num_onmf_components}_repeat_{cur_seed}.npy', allow_pickle=True).item()
            rec_components[ii] = cur_onmf.components_

        all_components = rec_components.reshape(-1, np.sum(prior_programs_mask))

        gene_group_ind = summary_components(all_components, num_onmf_components, summary_method=summary_method, figure_folder=self.fig_folder)

        # Mask the prior programs and add them to the gene group indices
        sub_mask_ind = np.where(prior_programs_mask)[0]
        gene_group_ind = [sub_mask_ind[gene_list] for gene_list in gene_group_ind]
        gene_group_ind += self.prior_programs_ind

        # Initialize and compute the summary matrix
        components_summary = np.zeros((num_spin, adata.shape[1]))
        for ii in range(num_spin):
            sub_matrix = large_subsample_matrix[:, gene_group_ind[ii]]
            sub_onmf = NMF(n_components=1, init='random', random_state=0).fit(sub_matrix)
            components_summary[ii, gene_group_ind[ii]] = sub_onmf.components_

        # Normalize the summary components and finalize the onmf_summary object
        components_summary = normalize(components_summary, axis=1, norm='l2')
        onmf_summary = NMF(n_components=num_spin, init='random', random_state=0)
        onmf_summary.components_ = components_summary

        return onmf_summary

    def gene_program_discovery(self,
                               num_repeat: int = 10,
                               seed: int = 0,
                               cluster_key: str = 'leiden',
                               mode: str = 'compute_summary',
                               prior_programs: List[List[str]] = None,
                               params: dict = {}) -> None:
        """
        Discover gene programs by performing oNMF decomposition on the annotated data.

        Parameters
        ----------
        num_onmf_components : int, optional
            Number of oNMF components. Default is None.
        num_subsample : int, optional
            Number of samples to use after subsampling. Default is 10000.
        num_subsample_large : int, optional
            Number of samples for large subsampling. Default is None.
        num_repeat : int, optional
            Number of times to repeat oNMF. Default is 10.
        seed : int, optional
            Random seed for reproducibility. Default is 0.
        balance_obs : str, optional
            Observation category to balance. Default is None.
        balance_method : str, optional
            Method used for balancing categories. Default is None.
        max_sample_rate : float, optional
            Maximum sampling rate. Default is 2.
        prior_programs : List[List[str]], optional
            List of predefined gene programs. Default is None.
        summary_method : str, optional
            Method to summarize oNMF components. Default is 'kmeans'.
        """

        if mode not in ['compute_summary', 'compute_only', 'summary_only']:
            raise ValueError(
                'Mode must be one of compute_summary, compute_only, or summary_only')
        
        adata = self.adata
        num_spin = self.num_spin

        onmf_parameters = {'num_onmf_components': num_spin,
                           'num_subsample': 10000,
                           'num_subsample_large': None,
                           'balance_method': 'squareroot',
                           'max_sample_rate': 2,
                           'summary_method': 'kmeans',
                           'std_clip_percentile': 20,
                           'min_cluster_size': 20,
                           'onmf_epoch_number': 500,
                           'seed_list': seed + np.arange(num_repeat)}
        
        onmf_parameters.update(params)
        
        if onmf_parameters['num_subsample_large'] is None:
            onmf_parameters['num_subsample_large'] = min(onmf_parameters['num_subsample'] * 5, adata.shape[0])

        if onmf_parameters['balance_method'] not in ['equal', 'proportional', 'squareroot', None]:
            raise ValueError('balance_method must be one of equal, proportional, squareroot, or None')

        if onmf_parameters['summary_method'] not in ['kmeans', 'leiden']:
            raise ValueError('summary_method must be one of kmeans or leiden')

        # Process prior programs if provided and validate against num_spin
        if prior_programs is not None:
            prior_program_ind = [[np.where(adata.var_names == gene)[0][0] for gene in gene_list] for gene_list in prior_programs]
            preprogram_flat = [gene for program in prior_programs for gene in program]
            if len(prior_programs) > num_spin:
                raise ValueError('Number of prior_programs must be less than the number of spins')
            # Generate mask to exclude genes in prior programs from the annotated data
            prior_program_mask = ~ np.isin(adata.var_names, preprogram_flat)
            onmf_parameters['num_onmf_components'] -= len(prior_programs)
        else:
            prior_program_mask = np.ones(adata.shape[1], dtype=bool)
            prior_program_ind = []

        # number of oNMF components = number of spins - number of prior programs
        self.onmf_parameters = onmf_parameters 
        self.prior_programs_mask = prior_program_mask
        self.prior_programs_ind = prior_program_ind

        num_onmf_components = onmf_parameters['num_onmf_components']
        
        if mode == 'compute_summary' or mode == 'compute_only':

            os.makedirs(self.save_path + 'onmf/', exist_ok=True)

            cur_gene_matrix = adata.X[:, prior_program_mask]
            cluster_label_raw = adata.obs[cluster_key].values
            cluster_label = pd.factorize(cluster_label_raw)[0]

            for rep in range(num_repeat):
                
                cur_seed = onmf_parameters['seed_list'][rep]

                std_clipped, sub_gene_matrix_normed = subsample_normalize_gene_matrix(cur_gene_matrix, cluster_label, num_subsample=onmf_parameters['num_subsample'], seed=cur_seed, params=onmf_parameters)

                compute_onmf_decomposition(sub_gene_matrix_normed, num_onmf_components, seed=cur_seed, params=onmf_parameters, save_path=self.save_path + f'onmf/onmf_components_{num_onmf_components}_repeat_{cur_seed}.npy')
            
        if mode == 'compute_summary' or mode == 'summary_only':

            # Perform subsampling and standard deviation clipping on the matrix
            std_clipped, sub_large_gene_matrix_normed = subsample_normalize_gene_matrix(cur_gene_matrix, cluster_label, num_subsample=onmf_parameters['num_subsample_large'], seed=seed + rep, params=onmf_parameters)
            self.matrix_std = std_clipped
            
            # Summarize the oNMF decompositions
            onmf_summary = self.summarize_onmf_result(sub_large_gene_matrix_normed, num_onmf_components, num_repeat)
            
            self._onmf_summary = onmf_summary

            # Save the gene programs to a CSV file
            file_path = onmf_to_csv(onmf_summary.components_, adata.var_names, self.save_path + f'gene_programs_total_{num_spin}_onmf_{num_onmf_components}_prior_{len(prior_program_ind)}.csv', thres=0.01)
            print('Gene programs saved to {}'.format(file_path))
            self.gene_program_csv = file_path

            gene_matrix = self.adata.X
            if issparse(gene_matrix):
                gene_matrix = np.asarray(gene_matrix.toarray()).astype(np.float64)
            # Transform the original matrix by the oNMF summary components and normalize by standard deviation
            self._onmf_rep_ori = onmf_summary.transform(
                gene_matrix / self.matrix_std)

            self.discretize()


class DSPIN(object):
    """
    DSPIN class determines the appropriate subclass (GeneDSPIN or ProgramDSPIN)
    based on the dataset's gene count and spin count.
    """

    def __new__(cls,
                adata: ad.AnnData,
                save_path: str,
                num_spin: int = None,
                filter_threshold: float = 0.02,
                **kwargs) -> Union[GeneDSPIN, ProgramDSPIN]:
        """
        Initialize DSPIN and select the appropriate subclass.

        Parameters
        ----------
        adata : ad.AnnData
            The annotated data matrix (n_cells x n_genes).
        save_path : str
            Path to save results.
        num_spin : int, optional
            Number of spins for network inference. Default is 15 or number of genes if lower.
        filter_threshold : float, optional
            Minimum fraction of cells expressing a gene for it to be retained. Default is 0.02.
        **kwargs : dict
            Additional keyword arguments passed to the selected subclass.

        Returns
        -------
        Union[GeneDSPIN, ProgramDSPIN]
            An instance of either GeneDSPIN or ProgramDSPIN based on gene count.
        """

        if adata.X.min() < 0:
            warnings.warn("Negative expression detected. Expect normalized log-transformed data.")

        # Filtering genes based on the filter_threshold.
        sc.pp.filter_genes(adata, min_cells=adata.shape[0] * filter_threshold)
        print(
            '{} genes have expression in more than {} of the cells'.format(
                adata.shape[1],
                filter_threshold))

        # if the number of genes is less than the number of spins, use GeneDSPIN
        if num_spin is None:
            num_spin = np.min([15, adata.shape[1]])

        if adata.shape[1] == num_spin:
            return GeneDSPIN(adata, save_path, num_spin=num_spin, **kwargs)
        elif adata.shape[1] < num_spin:
            warnings.warn(
                "Number of genes is less than the number of spins. Using number of genes as num_spin.")
            num_spin = adata.shape[1]
            return GeneDSPIN(adata, save_path, num_spin=num_spin, **kwargs)
        # otherwise, use ProgramDSPIN because oNMF is needed
        else:
            return ProgramDSPIN(adata, save_path, num_spin=num_spin, **kwargs)


