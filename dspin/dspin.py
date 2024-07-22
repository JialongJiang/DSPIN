# -*-coding:utf-8 -*-
'''
@Time    :   2023/10/13 10:43
@Author  :   Jialong Jiang, Yingying Gong
'''

from .plot import (
    onmf_to_csv
)
from .compute import (
    onmf_discretize,
    sample_corr_mean,
    learn_network_adam,
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
from typing import List


class AbstractDSPIN(ABC):
    """
    Parent and abstract class for DSPIN, not to be instantiated directly.
    Contains methods and properties common to both GeneDSPIN and ProgramDSPIN.

    ...
    Attributes
    ----------
    adata : ad.AnnData
        Annotated data.
    save_path : str
        Path where results will be saved.
    num_spin : int
        Number of spins.

    Methods
    -------
    discretize()
        Discretizes the ONMF representation into three states (-1, 0, 1) using K-means clustering.
    raw_data_corr(sample_col_name)
        Computes correlation of raw data based on sample column name.
    raw_data_state(sample_col_name)
        Calculate and return the correlation of raw data.
    default_params(method)
        Provide the default parameters for the specified algorithm.
    network_inference(sample_col_name, method, params, example_list, record_step)
        Execute the network inference using the specified method and parameters and record the results.
    """

    def __init__(self,
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int):
        """
        Initialize the AbstractDSPIN object with specified Annotated Data, save path, and number of spins.

        Parameters:
        adata (ad.AnnData): Annotated Data.
        save_path (str): Path where results will be saved.
        num_spin (int): number of spins.
        """

        self.adata = adata
        self.save_path = os.path.abspath(save_path) + '/'
        self.num_spin = num_spin

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print("Saving path does not exist. Creating a new folder.")
        self.fig_folder = self.save_path + 'figs/'
        os.makedirs(self.fig_folder, exist_ok=True)

    @property
    def program_representation(self):
        return self._onmf_rep_ori

    @property
    def program_discretized(self):
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

    @program_representation.setter
    def program_representation(self, value):
        self._onmf_rep_ori = value

    @program_discretized.setter
    def program_discretized(self, value):
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

    def discretize(self) -> np.ndarray:
        """
        Discretizes the ONMF representation into three states (-1, 0, 1) using K-means clustering.

        Returns:
        - np.ndarray: The discretized ONMF representation.
        """
        onmf_rep_ori = self.program_representation
        fig_folder = self.fig_folder

        onmf_rep_tri = onmf_discretize(onmf_rep_ori, fig_folder)
        self._onmf_rep_tri = onmf_rep_tri

    def raw_data_corr(self, sample_col_name) -> np.ndarray:
        """
        Computes correlation of raw data based on sample column name.

        :param sample_col_name: The name of the sample column to be used for correlation computation.
        :return: The correlated raw data.
        """

        raw_data, samp_list = sample_corr_mean(
            self.adata.obs[sample_col_name], self._onmf_rep_tri)
        self._raw_data = raw_data
        self._samp_list = samp_list

    def raw_data_state(self, sample_col_name) -> np.ndarray:
        """
        Calculate and return the correlation of raw data.

        Parameters:
        sample_col_name (str): The name of the column in the sample to calculate the correlation.

        Returns:
        np.ndarray: Array representing the correlated raw data.
        """
        cadata = self.adata
        onmf_rep_tri = self._onmf_rep_tri

        samp_list = np.unique(cadata.obs[sample_col_name])
        state_list = np.zeros(len(samp_list), dtype=object)

        for ii, cur_samp in enumerate(samp_list):
            cur_filt = cadata.obs[sample_col_name] == cur_samp
            cur_state = self._onmf_rep_tri[cur_filt, :]
            state_list[ii] = cur_state.T

        self._raw_data = state_list
        self._samp_list = samp_list

    def default_params(self,
                       method: str) -> dict:
        """
        Provide the default parameters for the specified algorithm.

        Parameters:
        method (str): The method for which to return the default parameters.

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
                  'save_path': self.save_path}
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
                          sample_col_name: str = 'sample_id',
                          method: str = 'auto',
                          directed: bool = False,
                          params: dict = None,
                          example_list: List[str] = None,
                          record_step: int = 10,
                          run_with_matlab: bool = False, 
                          precomputed_discretization: np.array = None):
        """
        Execute the network inference using the specified method and parameters and record the results.

        Parameters:
        sample_col_name (str): The name of the sample column.
        method (str, optional): The method used for network inference, default is 'auto'.
        params (dict, optional): Dictionary of parameters to be used, default is None.
        example_list (List[str], optional): List of examples to be used, default is None.
        record_step (int, optional): The step interval to record the results, default is 10.

        Raises:
        ValueError: If an invalid method is specified.
        """

        self.sample_col_name = sample_col_name

        if method not in [
            'maximum_likelihood',
            'mcmc_maximum_likelihood',
            'pseudo_likelihood',
            'directed_pseudo_likelihood',
                'auto']:
            raise ValueError(
                "Method must be one of 'maximum_likelihood', 'mcmc_maximum_likelihood', 'pseudo_likelihood' or 'auto'.")

        if method == 'auto':           
            samp_list = np.unique(self.adata.obs[sample_col_name])
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

        if precomputed_discretization is not None:
            self._onmf_rep_tri = precomputed_discretization

        if method == 'pseudo_likelihood':
            self.raw_data_state(sample_col_name)
        else:
            self.raw_data_corr(sample_col_name)

        self.example_list = example_list

        train_dat = self.default_params(method)
        train_dat['rec_gap'] = record_step
        train_dat['directed'] = directed
        if params is not None:
            train_dat.update(params)

        if run_with_matlab:
            if method in ['maximum_likelihood', 'mcmc_maximum_likelihood']:
                file_path = self.save_path + 'raw_data.mat'
            elif method in ['pseudo_likelihood', 'directed_pseudo_likelihood']:
                file_path = self.save_path + 'raw_data_state.mat'
            savemat(file_path, {'raw_data': self._raw_data, 'sample_list': self._samp_list, **train_dat})
            print("Data saved to {}. Please run the network inference in MATLAB and load the results back.".format(file_path))

        else:
            cur_j, cur_h = learn_network_adam(self.raw_data, method, train_dat)
            self._network = cur_j
            self._responses = cur_h

    def response_relative_to_control(self, 
                           if_control: np.array, 
                           batch_index: np.array):
        """
        Compute the relative responses based on the control samples in each batch or all batches.

        Parameters:
        if_control (numpy.ndarray): A boolean array indicating which samples are control samples.
        batch_index (numpy.ndarray): An array indicating the batch assignment for each sample.
        """

        self._relative_responses = compute_relative_responses(cur_h, if_control, batch_index)


class GeneDSPIN(AbstractDSPIN):
    """
    GeneDSPIN inherits the AbstractDSPIN class, optimized for handling smaller datasets.
    It doesn't need ONMF and can directly perform network inference on the genes.
    """

    def __init__(self,
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int):
        """
        Initialize the GeneDSPIN object.

        Parameters:
        adata (ad.AnnData): annotated data, contains observed data with annotations.
        save_path (str): Path where results will be saved.
        num_spin (int): Specifies the number of spins.
        """

        super().__init__(adata, save_path, num_spin)
        print("GeneDSPIN initialized.")

        if issparse(adata.X):
            self._onmf_rep_ori = np.array(adata.X)
        else:
            self._onmf_rep_ori = adata.X

        self.discretize()


class ProgramDSPIN(AbstractDSPIN):
    """
    ProgramDSPIN inherits the AbstractDSPIN class, optimized for handling larger datasets.
    It contains specialized methods to handle subsampling and balancing of large gene matrices.

    ...
    Attributes
    ----------
    adata : ad.AnnData
        Annotated data.
    save_path : str
        Path where results will be saved.
    num_spin : int
        Number of spins.
    num_onmf_components : int
        Number of ONMF components.
    preprograms : List[List[str]]
        List of preprograms.
    num_repeat : int
        Number of times to repeat ONMF.

    Methods
    -------
    subsample_matrix_balance(total_sample_size, std_clip_percentile)
        Subsample and balance a gene matrix to achieve the desired sample size and standard deviation clipping percentile.
    compute_onmf_repeats(num_onmf_components, num_repeat, num_subsample, seed, std_clip_percentile)
        Compute ONMF decompositions repeatedly after subsampling of the gene matrix for each repetition.
    summarize_onmf_result(num_onmf_components, num_repeat, summary_method)
        Summarize the results of the repeated ONMF decompositions, integrating the components obtained from each ONMF decomposition and preprograms if provided.
    gene_program_discovery(num_onmf_components, num_subsample, num_subsample_large, num_repeat, balance_obs, balance_method, max_sample_rate, prior_programs, summary_method)
        Discovers gene programs by performing ONMF decomposition on the given annotated data object.
    """

    def __init__(self,
                 adata: ad.AnnData,
                 save_path: str,
                 num_spin: int,
                 num_onmf_components: int = None,
                 preprograms: List[List[str]] = None,
                 num_repeat: int = 10):
        """
        Initialize the ProgramDSPIN object.

        Parameters:
        adata (ad.AnnData): An annotated data matrix, contains observed data with annotations.
        save_path (str): Path where results will be saved.
        num_spin (int): Specifies the number of spins.
        num_onmf_components (int, optional): The number of ONMF components, default is None.
        preprograms (List[List[str]], optional): List of preprograms, default is None.
        num_repeat (int, optional): The number of times to repeat, default is 10.
        """
        super().__init__(adata, save_path, num_spin)

        print("ProgramDSPIN initialized.")

        self._onmf_summary = None
        self._gene_matrix_large = None
        self._use_data_list = None
        self.gene_program_csv = None
        self.preprograms = None
        self.preprogram_num = len(preprograms) if preprograms else 0

    @property
    def onmf_decomposition(self):
        return self._onmf_summary

    @onmf_decomposition.setter
    def onmf_decomposition(self, value):
        self._onmf_summary = value

    def subsample_matrix_balance(self,
                                 total_sample_size: int,
                                 std_clip_percentile: float = 20) -> (np.ndarray,
                                                                      np.ndarray):
        """
        Subsample and balance a gene matrix to achieve the desired sample size and
        standard deviation clipping percentile.

        Parameters:
        total_sample_size (int): The desired sample size after subsampling.
        std_clip_percentile (float, optional): The desired percentile for standard deviation clipping,
                                               default is 20.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the clipped standard deviations
                                       and the balanced and normalized gene matrix.
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
                    gene_matrix_balanced[strart_ind: end_ind,
                                         :] = cadata.X[cur_filt,
                                                       :][sele_ind,
                                                          :].toarray()
                else:
                    gene_matrix_balanced[strart_ind: end_ind,
                                         :] = cadata.X[cur_filt, :][sele_ind, :]

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
                             std_clip_percentile: float = 20):
        """
        Compute ONMF decompositions repeatedly after subsampling of the gene matrix for each repetition.

        Parameters:
        num_onmf_components (int): The number of ONMF components.
        num_repeat (int): The number of times to repeat the ONMF computation.
        num_subsample (int): The number of samples to obtain after subsampling.
        seed (int, optional): Seed for reproducibility. Default is 0.
        std_clip_percentile (float, optional): Percentile for standard deviation clipping. Default is 20.
        """

        preprograms = self.prior_programs
        adata = self.adata
        
        # Create the directory for saving ONMF decompositions if it doesn't
        # exist
        os.makedirs(self.save_path + 'onmf/', exist_ok=True)

        print("Computing ONMF decomposition...")

        # Repeat ONMF decomposition for num_repeat times
        for ii in range(num_repeat):
            np.random.seed(seed + ii)
            # Check if ONMF decomposition for the current iteration exists
            if os.path.exists(
                    self.save_path + 'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii)):
                print(
                    "ONMF decomposition {} already exists. Skipping...".format(ii))
                continue

            # Subsample the matrix and normalize it
            _, gene_matrix_norm = self.subsample_matrix_balance(
                num_subsample, std_clip_percentile=std_clip_percentile)

            # Compute current ONMF decomposition and save the result
            current_onmf = compute_onmf(
                seed + ii, num_onmf_components, gene_matrix_norm[:, self.prior_programs_mask])
            np.save(self.save_path +
                    'onmf/onmf_rep_{}_{}.npy'.format(num_onmf_components, ii), current_onmf)

    def summarize_onmf_result(self,
                              num_onmf_components: int,
                              num_repeat: int,
                              summary_method: str):
        """
        Summarize the results of the repeated ONMF decompositions,
        integrating the components obtained from each ONMF decomposition and preprograms if provided.

        Parameters:
        num_onmf_components (int): The number of ONMF components.
        num_repeat (int): The number of repetitions for ONMF computation.
        summary_method (str): The method used for summarizing the components.

        Returns:
        object: An NMF object containing the summarized components.
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
                'onmf/onmf_rep_{}_{}.npy'.format(
                    num_onmf_components,
                    ii),
                allow_pickle=True).item()
            rec_components[ii] = cur_onmf.components_

        all_components = rec_components.reshape(-1,
                                                np.sum(prior_programs_mask))

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
        Discovers gene programs by performing ONMF decomposition on the given annotated data object.

        Parameters:
        num_onmf_components (int, optional): The number of ONMF components. Default is None.
        num_subsample (int, optional): Number of samples to obtain after subsampling. Default is 10000.
        num_subsample_large (int, optional): Number of samples for large subsampling. Default is None.
        num_repeat (int, optional): Number of times to repeat the ONMF computation. Default is 10.
        balance_obs (str, optional): Observation to balance. Default is None.
        balance_method (str, optional): Method used for balancing. Default is None.
        max_sample_rate (float, optional): Maximum sample rate. Default is 2.
        prior_programs (List[List[str]], optional): List of prior programs. Default is None.
        summary_method (str, optional): Method used for summarizing the components. Default is 'kmeans'.

        Raises:
        ValueError: If balance_method is not one of equal, proportional, squareroot, or None
        ValueError: If summary_method is not one of kmeans or leiden
        ValueError: If the number of prior_programs is greater than the number of spins
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
                    'Number of preprograms must be less than the number of spins')

            # Generate mask to exclude genes in prior programs from the
            # annotated data
            prior_program_mask = ~ np.isin(adata.var_names, preprogram_flat)
        else:
            prior_program_mask = np.ones(adata.shape[1], dtype=bool)
            prior_program_ind = []

        # number of ONMF components = number of spins - number of prior
        # programs
        if num_onmf_components is None:
            num_onmf_components = self.num_spin - len(prior_program_ind)

        self.prior_programs_mask = prior_program_mask
        self.prior_programs_ind = prior_program_ind

        np.random.seed(seed)
        # Perform subsampling and standard deviation clipping on the matrix
        self.matrix_std, self.large_subsample_matrix = self.subsample_matrix_balance(
            num_subsample_large, std_clip_percentile=20)

        # Compute ONMF decompositions repeatedly
        self.compute_onmf_repeats(
            num_onmf_components,
            num_repeat,
            num_subsample,
            seed,
            std_clip_percentile=20)

        # Summarize the ONMF decompositions
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
        # Transform the original matrix by the ONMF summary components and
        # normalize by standard deviation
        self._onmf_rep_ori = onmf_summary.transform(
            gene_matrix / self.matrix_std)

        self.discretize()


class DSPIN(object):
    """
    DSPIN class client end.

    This class serves as a automatic conditional constructor to decide
    which subclass of DSPIN to instantiate based on the number of genes
    and spins provided.
    """

    def __new__(cls,
                adata: ad.AnnData,
                save_path: str,
                num_spin: int = None,
                filter_threshold: float = 0.02,
                **kwargs):
        """
        Initialize the DSPIN object.

        Parameters:
        adata (ad.AnnData): The annotated data of shape n_cells x n_genes.
        save_path (str): The path to save any output or results.
        num_spin (int, optional): The number of spins. Defaults to 15.
        filter_threshold (float, optional): The threshold to filter genes. Defaults to 0.02.
        **kwargs: Keyword arguments, see classes GeneDSPIN and ProgramDSPIN for more details.

        Returns:
        DSPIN object: An instance of either GeneDSPIN or ProgramDSPIN subclass.
        """

        # Filtering genes based on the filter_threshold.
        # It retains only those genes that have expression
        # in more than the specified percentage of cells.
        sc.pp.filter_genes(adata, min_cells=adata.shape[0] * filter_threshold)
        print(
            '{} genes have expression in more than {} of the cells'.format(
                adata.shape[1],
                filter_threshold))

        # if the number of genes is less than the number of spins, use
        # GeneDSPIN
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
