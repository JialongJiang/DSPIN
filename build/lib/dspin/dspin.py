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
    category_balance_number,
    compute_onmf,
    summary_components,
    compute_relative_responses
)
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
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
        params = {'num_epoch': 200,
                  'cur_j': np.zeros((num_spin, num_spin)),
                  'cur_h': np.zeros((num_spin, num_sample)),
                  'save_path': self.save_path, 
                  'rec_gap': 10}
        params.update({'lambda_l1_j': 0.01,
                       'lambda_l1_h': 0,
                       'lambda_l2_j': 0,
                       'lambda_l2_h': 0.005})
        params.update({'backtrack_gap': 20,
                       'backtrack_tol': 4})

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

    def subsample_matrix_balance(self,
                                 total_sample_size: int,
                                 std_clip_percentile: float = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample and balance the gene matrix based on desired sample size and standard deviation clipping.

        Parameters
        ----------
        total_sample_size : int
            The target number of cells after subsampling.
        std_clip_percentile : float, optional
            Percentile for standard deviation clipping, default is 20.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Clipped standard deviations and the balanced, normalized gene matrix.
        """
        gene_matrix = self.adata.X
        cadata = self.adata
        num_cell, num_gene = gene_matrix.shape

        balance_by = self.balance_obs
        method = self.balance_method
        maximum_sample_rate = self.max_sample_rate

        # If no balancing is required, simply subsample the matrix
        if method is None:
            gene_matrix_balanced = gene_matrix[np.random.choice(
                num_cell, total_sample_size, replace=False), :]
        else:
            cluster_list = list(cadata.obs[balance_by].value_counts().keys())
            cluster_count = list(cadata.obs[balance_by].value_counts())

            sampling_number = category_balance_number(
                total_sample_size, cluster_count, method, maximum_sample_rate)

            gene_matrix_balanced = np.zeros(
                (np.sum(sampling_number), cadata.X.shape[1]))

            for ii in range(len(cluster_list)):
                cur_num = sampling_number[ii]
                cur_filt = cadata.obs[balance_by] == cluster_list[ii]
                sele_ind = np.random.choice(np.sum(cur_filt), cur_num)
                strart_ind = np.sum(sampling_number[:ii])
                end_ind = strart_ind + cur_num
                if issparse(cadata.X):
                    gene_matrix_balanced[strart_ind: end_ind, :] = cadata.X[cur_filt, :][sele_ind, :].toarray()
                else:
                    gene_matrix_balanced[strart_ind: end_ind, :] = cadata.X[cur_filt, :][sele_ind, :]

        # Normalize the matrix by standard deviation
        if issparse(gene_matrix_balanced):
            gene_matrix_balanced = np.asarray(gene_matrix_balanced.toarray())

        std = gene_matrix_balanced.std(axis=0)
        std_clipped = std.clip(np.percentile(std, std_clip_percentile), np.inf)
        gene_matrix_balanced_normalized = gene_matrix_balanced / std_clipped

        return std_clipped, gene_matrix_balanced_normalized

    def compute_onmf_repeats(self,
                             num_onmf_components: int,
                             num_repeat: int,
                             num_subsample: int,
                             seed: int = 0,
                             std_clip_percentile: float = 20) -> None:
        """
        Perform multiple oNMF decompositions on subsampled data to enhance stability.

        Parameters
        ----------
        num_onmf_components : int
            Number of oNMF components.
        num_repeat : int
            Number of repetitions for oNMF computation.
        num_subsample : int
            Number of cells to use while computing the oNMF
        seed : int, optional
            Seed for reproducibility. Default is 0.
        std_clip_percentile : float, optional
            Percentile for standard deviation clipping. Default is 20.
        """

        prior_programs = self.prior_programs
        adata = self.adata
        
        # Create the directory for saving oNMF decompositions if it doesn't
        # exist
        os.makedirs(self.save_path + 'onmf/', exist_ok=True)

        print("Computing oNMF decomposition...")

        # Repeat oNMF decomposition for num_repeat times
        for ii in range(num_repeat):
            np.random.seed(seed + ii)
            # Check if oNMF decomposition for the current iteration exists
            if os.path.exists(
                    self.save_path + 'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii)):
                print(
                    "oNMF decomposition {} already exists. Skipping...".format(ii))
                continue

            # Subsample the matrix and normalize it
            _, gene_matrix_norm = self.subsample_matrix_balance(
                num_subsample, std_clip_percentile=std_clip_percentile)

            # Compute current oNMF decomposition and save the result
            current_onmf = compute_onmf(
                seed + ii, num_onmf_components, gene_matrix_norm[:, self.prior_programs_mask])
            np.save(self.save_path +
                    'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii), current_onmf)

    def summarize_onmf_result(self,
                              num_onmf_components: int,
                              num_repeat: int,
                              summary_method: str) -> NMF:
        """
        Summarize the results of repeated oNMF decompositions,
        integrating components from each oNMF run and prior programs if available.

        Parameters
        ----------
        num_onmf_components : int
            Number of oNMF components.
        num_repeat : int
            Number of oNMF repetitions.
        summary_method : str
            The method used for summarizing oNMF components.

        Returns
        -------
        NMF
            An NMF object containing the summarized components.
        """

        # Retrieve the number of spins and annotated data
        num_spin = self.num_spin
        adata = self.adata
        prior_programs_mask = self.prior_programs_mask

        rec_components = np.zeros(
            (num_repeat, num_onmf_components, np.sum(prior_programs_mask)))

        for ii in range(num_repeat):
            cur_onmf = np.load(
                self.save_path +
                'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii),
                allow_pickle=True).item()
            rec_components[ii] = cur_onmf.components_

        all_components = rec_components.reshape(-1, np.sum(prior_programs_mask))

        gene_group_ind = summary_components(
            all_components,
            num_onmf_components,
            num_repeat,
            summary_method=summary_method)

        # Mask the prior programs and add them to the gene group indices
        sub_mask_ind = np.where(prior_programs_mask)[0]
        gene_group_ind = [sub_mask_ind[gene_list]
                          for gene_list in gene_group_ind]
        gene_group_ind += self.prior_programs_ind

        # Initialize and compute the summary matrix
        components_summary = np.zeros((num_spin, adata.shape[1]))
        for ii in range(num_spin):
            sub_matrix = self.large_subsample_matrix[:, gene_group_ind[ii]]
            sub_onmf = NMF(
                n_components=1,
                init='random',
                random_state=0).fit(sub_matrix)
            components_summary[ii, gene_group_ind[ii]] = sub_onmf.components_

        # Normalize the summary components and finalize the onmf_summary object
        components_summary = normalize(components_summary, axis=1, norm='l2')
        onmf_summary = NMF(
            n_components=num_spin,
            init='random',
            random_state=0)
        onmf_summary.components_ = components_summary

        return onmf_summary

    def gene_program_discovery(self,
                               num_onmf_components: int = None,
                               num_subsample: int = 10000,
                               num_subsample_large: int = None,
                               num_repeat: int = 10,
                               seed: int = 0,
                               balance_obs: str = None,
                               balance_method: str = None,
                               max_sample_rate: float = 2,
                               prior_programs: List[List[str]] = None,
                               summary_method: str = 'kmeans'):
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

        if balance_method not in ['equal', 'proportional', 'squareroot', None]:
            raise ValueError(
                'balance_method must be one of equal, proportional, squareroot, or None')

        if summary_method not in ['kmeans', 'leiden']:
            raise ValueError('summary_method must be one of kmeans or leiden')

        adata = self.adata
        num_spin = self.num_spin

        if num_subsample_large is None:
            num_subsample_large = min(num_subsample * 5, self.adata.shape[0])

        # Set default balance method if balance_obs is provided but
        # balance_method is None
        if (balance_obs is not None) & (balance_method is None):
            balance_method = 'squareroot'

        self.balance_obs = balance_obs
        self.balance_method = balance_method
        self.max_sample_rate = max_sample_rate
        self.prior_programs = prior_programs

        # Process prior programs if provided and validate against num_spin
        if prior_programs is not None:
            prior_program_ind = [
                [np.where(adata.var_names == gene)[0][0] for gene in gene_list]
                for gene_list in prior_programs]
            preprogram_flat = [
                gene for program in prior_programs for gene in program]
            if len(prior_programs) > num_spin:
                raise ValueError(
                    'Number of prior_programs must be less than the number of spins')

            # Generate mask to exclude genes in prior programs from the
            # annotated data
            prior_program_mask = ~ np.isin(adata.var_names, preprogram_flat)
        else:
            prior_program_mask = np.ones(adata.shape[1], dtype=bool)
            prior_program_ind = []

        # number of oNMF components = number of spins - number of prior
        # programs
        if num_onmf_components is None:
            num_onmf_components = self.num_spin - len(prior_program_ind)

        self.prior_programs_mask = prior_program_mask
        self.prior_programs_ind = prior_program_ind

        np.random.seed(seed)
        # Perform subsampling and standard deviation clipping on the matrix
        self.matrix_std, self.large_subsample_matrix = self.subsample_matrix_balance(
            num_subsample_large, std_clip_percentile=20)

        # Compute oNMF decompositions repeatedly
        self.compute_onmf_repeats(
            num_onmf_components,
            num_repeat,
            num_subsample,
            seed,
            std_clip_percentile=20)

        # Summarize the oNMF decompositions
        onmf_summary = self.summarize_onmf_result(
            num_onmf_components, num_repeat, summary_method)
        self._onmf_summary = onmf_summary

        # Save the gene programs to a CSV file
        file_path = onmf_to_csv(
            onmf_summary.components_,
            adata.var_names,
            self.save_path +
            'gene_programs_{}_{}_{}.csv'.format(
                len(prior_program_ind),
                num_onmf_components,
                num_spin),
            thres=0.01)
        print('Gene programs saved to {}'.format(file_path))
        self.gene_program_csv = file_path

        gene_matrix = self.adata.X
        if issparse(gene_matrix):
            gene_matrix = np.asarray(gene_matrix.toarray()).astype(np.float64)
        # Transform the original matrix by the oNMF summary components and
        # normalize by standard deviation
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
