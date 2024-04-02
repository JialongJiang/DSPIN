# -*-coding:utf-8 -*-
'''
@Time    :   2024/04/02 14:29
@Author  :   Jialong Jiang
'''


import csv
import pandas as pd
from typing import List
import numpy as np
import os
import logging

# import openai

system_question = "Your are a desperate biology phd professor studying gene expression from a dataset of cancer patients. Given following a list of genes in the gene program, you have two tasks: \
    Firstly, summarize functions of the program in a few important functions. \
        Secondly, give a higher-level name of the gene program based on its function. You are required by your mentor to name it as short as possible.\
            You get 100 score if you can name it in two or three words, 80 score if you can name it in four words, 20 score for five or more words. \
                A limitation to this general rule is that you will lose 20 score if you name a gene program similar 'Gene Expression Program' or 'Gene Regulation Progran'. \
                    Please also avoid 'cancer' in the name you give. \
                        The results should be in the format: name: (your answer) \n function 1: (), related genes: ();\
                            function 2: (), related genes: ();\
                                ..... (more functions and their annotation, please don't list more than 10 functions)\
                                    You will be given a list of genes in the same gene program in each user input. "

def init_openai_api(api_key):
    openai.api_key = api_key

def gene_program_csv_to_list(file_path) -> List[str]:
    data = pd.read_csv(file_path)
    gene_programs = []
    for col in data.columns:
        column_values = data[col].dropna().tolist()
        gene_programs.append(','.join(column_values))
    return gene_programs


def generate_response(gene_program):
    messages = list()
    messages.append({"role": "system", "content": system_question})
    messages.append({"role": "user", "content": gene_program})
    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=messages, # The starting prompt for the AI
            max_tokens=1024,             # Adjust the desired response length
            temperature=0,            # Controls the randomness of the response
            n=1,                        # Number of responses to generate
        )
        # Get the response text from the API response
        response_text = response['choices'][0]['message']['content']
        return response_text
    except Exception as e:
        return str(e)
    
def generate_responses_to_csv(file_path, output_file_path):
    gene_programs = gene_program_csv_to_list(file_path)
    data = []
    responses = []
    for i in range(49,50):
        data_string = generate_response(gene_programs[i])
        responses.append(data_string)
        colon_index = data_string.find(':')
        newline_index = data_string.find('\n')
        name = data_string[colon_index + 1: newline_index]
        name = name.strip('Program')
        function = data_string[newline_index + 1:]
        data[i] = (i, name, function, gene_programs[i])
        print(str(i) + " program done" )
    
    with open(output_file_path, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'name', 'function', 'genes'])
        writer.writerows(data)


def automatic_david_annotation(data_folder, file_name, token_name, species='human'):
    """
    Main function to automate DAVID annotation for a set of genes.
    """
    # Validate input file extension
    assert file_name.endswith('.csv'), 'Input file must be a CSV file'

    try:
        import mygene
        from suds.client import Client
        import xlsxwriter
    except ImportError: 
        raise ImportError('Please install the required packages: mygene, suds, xlsxwriter')

    # Setup and read data
    all_onmf_df, all_gene = read_data(data_folder, file_name)
    num_list = all_onmf_df.shape[1]
    
    # Gene ID conversion
    gene_id_map = convert_gene_ids(all_gene, species)
    
    # Setup DAVID Web Service client
    client = setup_david_client(token_name)
    
    # Process each gene list
    process_gene_lists(data_folder, file_name, all_onmf_df, gene_id_map, client)

    gene_id_map_reverse = {v: k for k, v in gene_id_map.items()}
    
    # Save results to Excel
    save_to_excel(data_folder, file_name, gene_id_map_reverse, num_list)
    save_to_excel(data_folder, file_name, gene_id_map_reverse, num_list, full=True)


def read_data(data_folder, file_name):
    """
    Read gene data from a CSV file and return a unique list of gene names.
    
    Parameters:
        data_folder (str): The folder where the data file is located.
        file_name (str): The name of the CSV file containing gene data.
    
    Returns:
        list: A unique list of gene names.
    """
    program_file_name = os.path.join(data_folder, file_name)
    gene_df = pd.read_csv(program_file_name)
    gene_list = gene_df.values.flatten().astype('str')
    unique_genes = list(np.unique(gene_list))
    unique_genes.remove('nan')
    return gene_df, unique_genes


def convert_gene_ids(all_gene, species):
    """
    Convert gene symbols to ENSEMBL IDs using the MyGeneInfo service.
    
    Parameters:
        all_gene (list): A list of gene symbols.
        species (str): The species for gene ID conversion (default is 'human').
    
    Returns:
        dict: A dictionary mapping gene symbols to ENSEMBL IDs.
    """
    mg = mygene.MyGeneInfo()
    result = mg.querymany(all_gene, scopes="symbol", fields="ensembl.gene", species=species)
    gene_id_map = {}
    for gene_info in result:
        if 'ensembl' in gene_info:
            ensembl_info = gene_info['ensembl']
            ensembl_id = ensembl_info['gene'] if isinstance(ensembl_info, dict) else ensembl_info[0]['gene']
            gene_id_map[gene_info['query']] = ensembl_id
    return gene_id_map


def setup_david_client(token_name):
    """
    Setup and authenticate with the DAVID Web Service.
    
    Parameters:
        token_name (str): The authentication token for DAVID Web Service.
    
    Returns:
        Client: An authenticated suds client for DAVID Web Service.
    """
    url = 'https://david.ncifcrf.gov/webservice/services/DAVIDWebService?wsdl'
    client = Client(url)
    client.wsdl.services[0].setlocation('https://david.ncifcrf.gov/webservice/services/DAVIDWebService.DAVIDWebServiceHttpSoap11Endpoint/')
    client.service.authenticate(token_name)
    return client


def parse_david_output(term_clustering_report, save_path):
    """
    Parse the output from DAVID and save to a file.
    
    Parameters:
        term_clustering_report: The term clustering report from DAVID Web Service.
        save_path (str): The file path to save the parsed output.
    """
    total_clusters = len(term_clustering_report)
    print('Total clusters:', total_clusters)
    with open(save_path, 'w') as f_out:
        for i, cluster_record in enumerate(term_clustering_report, start=1):
            enrichment_score = cluster_record.score
            f_out.write(f'Annotation Cluster {i}\tEnrichmentScore:{enrichment_score}\n')
            headers = ['Category', 'Term', 'Count', '%', 'Pvalue', 'Genes', 'List Total', 'Pop Hits', 'Pop Total', 'Fold Enrichment', 'Bonferroni', 'Benjamini', 'FDR']
            f_out.write('\t'.join(headers) + '\n')
            for chart_record in cluster_record.simpleChartRecords:
                row = [
                    chart_record.categoryName,
                    chart_record.termName,
                    str(chart_record.listHits),
                    str(chart_record.percent),
                    str(chart_record.ease),
                    chart_record.geneIds,
                    str(chart_record.listTotals),
                    str(chart_record.popHits),
                    str(chart_record.popTotals),
                    str(chart_record.foldEnrichment),
                    str(chart_record.bonferroni),
                    str(chart_record.benjamini),
                    str(chart_record.afdr)
                ]
                f_out.write('\t'.join(row) + '\n')


def process_gene_lists(data_folder, file_name, all_onmf_df, gene_id_map, client):
    """
    Process each gene list, communicate with DAVID, and parse the results.

    Parameters:
        data_folder (str): The folder where the data files are located.
        file_name (str): The name of the original CSV file containing gene data.
        all_onmf_df (pd.DataFrame): A DataFrame containing gene lists.
        gene_id_map (dict): A dictionary mapping gene symbols to ENSEMBL IDs.
        client (Client): An authenticated suds client for DAVID Web Service.
    """
    
    def map_gene_ids(gene_id_map, gene_list):
        return gene_list.map(gene_id_map).fillna(gene_list)

    num_lists = all_onmf_df.shape[1]
    save_folder = os.path.join(data_folder, file_name.replace('.csv', '_david_results/'))
    os.makedirs(save_folder, exist_ok=True)

    for i in range(num_lists):
        cur_gene_list = all_onmf_df.iloc[:, i].dropna()
        cur_gene_list = map_gene_ids(gene_id_map, cur_gene_list)
        save_path = os.path.join(save_folder, f'list_{i}.csv')

        input_ids = ','.join(cur_gene_list)
        id_type = 'ENSEMBL_GENE_ID'
        list_name = f'List_{i}'
        list_type = 0

        client.service.addList(input_ids, id_type, list_name, list_type)
        term_clustering_report = client.service.getTermClusterReport(3, 3, 3, 0.5, 50)
        print(f'Gene list {i} ', end='\t')
        parse_david_output(term_clustering_report, save_path)


def save_to_excel(data_folder, file_name, gene_id_map_reverse, num_lists, full=True):
    """
    Save the parsed DAVID output to an Excel file, with an option to include full or filtered results.

    Parameters:
        data_folder (str): The folder where the data files are located.
        file_name (str): The name of the original CSV file containing gene data.
        gene_id_map_reverse (dict): A dictionary mapping ENSEMBL IDs to gene symbols.
        num_lists (int): The number of gene lists processed.
        full (bool): Whether to save the full results. If False, only results with Pvalue <= 0.05 are saved.
    """
    
    def map_gene_id_reverse(gene_name):
        try:
            return gene_id_map_reverse[gene_name]
        except:
            return gene_name
        
    def map_gene_id_reverse_list(gene_list):
        try:
            return ','.join([map_gene_id_reverse(gene) for gene in gene_list.split(', ')])
        except:
            return gene_list

    def format_pvalue(pvalue):
        try:
            if isinstance(pvalue, str):
                return '%.1e' % float(pvalue)
            else:
                return pvalue
        except:
            return pvalue

    def format_percent(value):
        try:
            if isinstance(value, str):
                return int(float(value))
            else:
                return value
        except:
            return value

    def filt_fun(pvalue):
        try:
            return float(pvalue) > 0.05
        except:
            return False

    save_folder = os.path.join(data_folder, file_name.replace('.csv', '_david_results/'))
    xlsx_name = file_name.replace('.csv', '_david_full.xlsx') if full else file_name.replace('.csv', '_david.xlsx')
    
    with pd.ExcelWriter(data_folder + xlsx_name) as writer:
        for ii in range(num_lists):
            cur_pd = pd.read_csv(save_folder + 'list_%d.csv' % ii, sep='\t', names=['Category', 'Term', 'Count', '%', 'Pvalue', 'Genes', 'List Total', 'Pop Hits', 'Pop Total', 'Fold Enrichment', 'Bonferroni', 'Benjamini', 'FDR'])
            filt1 = cur_pd['Category'] == 'UP_SEQ_FEATURE'
            filt2 = cur_pd['Pvalue'].apply(filt_fun)
            if not full:
                cur_pd = cur_pd[~(filt1 | filt2)]

            cur_pd['Term'] = cur_pd['Term'].apply(lambda x: x.split('~')[- 1])
            cur_pd['Genes'] = cur_pd['Genes'].apply(map_gene_id_reverse_list)
            cur_pd['Pvalue'] = cur_pd['Pvalue'].apply(format_pvalue)
            cur_pd['%'] = cur_pd['%'].apply(format_percent)

            ind_insert = np.where(cur_pd['Category'].str.startswith('Annotation Cluster'))[0]
            if len(ind_insert) > 1:
                for kk, ind in enumerate(ind_insert[1: ]):
                    cur_pd = pd.DataFrame(np.insert(cur_pd.values, ind + kk, values=[" "] * len(cur_pd.columns), axis=0),columns = cur_pd.columns)

            cur_pd.to_excel(writer, sheet_name='program_%d' % ii, index=False)
            writer.sheets['program_%d' % ii].set_column(0, 0, 25)
            writer.sheets['program_%d' % ii].set_column(1, 1, 50)
            writer.sheets['program_%d' % ii].set_column(2, 2, 5)
            writer.sheets['program_%d' % ii].set_column(3, 3, 5)
            writer.sheets['program_%d' % ii].set_column(5, 5, 100)

