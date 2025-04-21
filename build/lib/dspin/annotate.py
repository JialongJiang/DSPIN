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


def automatic_gpt_annotation(data_folder, file_name, gpt_model="gpt-4-0125-preview", temp=0.0, max_gene=30):
    """
    Main function to automate GPT annotation for a set of genes.
    """
    # Validate input file extension
    assert file_name.endswith('.csv'), 'Input file must be a CSV file'

    try:
        import openai
    except ImportError:
        raise ImportError('Please install the required packages: openai')

    # Setup and read data
    gene_df, all_gene = read_data(data_folder, file_name)
    num_list = gene_df.shape[1]

    output_dir = data_folder + file_name.replace('.csv', '_gpt/')
    context = "You are an efficient and insightful assistant to a molecular biologist"

    os.makedirs(output_dir, exist_ok=True)

    for ii in range(num_list):

        cur_gene_list = gene_df[str(ii)]
        cur_gene_list = cur_gene_list[~cur_gene_list.isna()][: max_gene]
        cur_prompt = make_user_prompt_with_score(cur_gene_list)

        response, fingerprint = openai_chat(
            context, cur_prompt, gpt_model, temp)

        output_file = output_dir + f'program_{ii}.txt'

        with open(output_file, 'w') as f:
            f.write(response)


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
        raise ImportError(
            'Please install the required packages: mygene, suds, xlsxwriter')

    # Setup and read data
    all_onmf_df, all_gene = read_data(data_folder, file_name)
    num_list = all_onmf_df.shape[1]

    # Gene ID conversion
    gene_id_map = convert_gene_ids(all_gene, species)

    # Setup DAVID Web Service client
    client = setup_david_client(token_name)

    # Process each gene list
    process_gene_lists(data_folder, file_name,
                       all_onmf_df, gene_id_map, client)

    gene_id_map_reverse = {v: k for k, v in gene_id_map.items()}

    # Save results to Excel
    save_to_excel(data_folder, file_name,
                    gene_id_map_reverse, all_onmf_df.columns.tolist(), full=False)
    save_to_excel(data_folder, file_name,
                    gene_id_map_reverse, all_onmf_df.columns.tolist(), full=True)


def make_user_prompt_with_score(genes, feature_df=[], direct=False, customized_prompt=None):
    """
    This function is adapted from "Evaluation of large language models for discovery of gene set function" 
    Arxiv: arxiv.org/abs/2309.04019v2 
    GitHub: github.com/idekerlab/llm_evaluation_for_gene_set_interpretation

    Create a "one shot" ChatGPT prompt based on the list of genes.
    :return: A string containing the ChatGPT prompt text
    """

    general_analysis_instructions = """
Be concise, do not use unnecessary words.
Be factual, do not editorialize.
Be specific, avoid overly general statements such as 'the proteins are involved in various cellular processes'.
Avoid listing facts about individual proteins. Instead, try to group proteins with similar functions and discuss their interplay, synergistyc or antagonistic effects and functional integration within the system.
Also avoid choosing generic process names such as 'Cellular Signaling and Regulation'.
If you cannot identify a prominent biological process for the proteins in the system, I want you to communicate this in you analysis and name the process: “System of unrelated proteins”. Provide a score of 0.00 for a "System of unrelated proteins".
    """

    task_instructions = """
Write a critical analysis of the biological processes performed by this system of interacting proteins.
Base your analysis on prior knowledge available in your training data.
After completing your analysis, propose a brief and detailed name for the most prominent biological process performed by the system.
    """

    score_instructions = """
After completing your analysis, please also assign a confidence score to the process name you selected.
This score should follow the name in parentheses and range from 0.00 to 1.00. A score of 0.00 indicates the lowest confidence,
while 1.00 reflects the highest confidence. This score helps gauge how accurately the chosen name represents the functions and activities
within the system of interacting proteins. When determining your score, consider the proportion of genes in the protein system that participate
in the identified biological process. For instance, if you select "Ribosome biogenesis" as the process name but only a few genes in the system 
contribute to this process, the score should be lower compared to a scenario where a majority of the genes are involved in "Ribosome biogenesis".
    """

    direct_instructions = """
Propose a name and provide analysis for the following gene set.
    """

    format_placeholder = """ 
Put your chosen name at the top of the analysis as 'Process: <name>’.
    """

    example_analysis = """
To help you in your work, I am providing an example system of interacting proteins and the corresponding example analysis output.

The example system of interacting proteins is:
PDX1, SLC2A2, NKX6-1, GLP1, GCG.

The example analysis output is:

Process: Pancreatic development and glucose homeostasis (0.96)

1. PDX1 is a homeodomain transcription factor involved in the specification of the early pancreatic epithelium and its subsequent differentiation. 
It activates the transcription of several genes including insulin, somatostatin, glucokinase and glucose transporter type 2. 
It is essential for maintenance of the normal hormone-producing phenotype in the pancreatic beta-cell. 
In pancreatic acinar cells, forms a complex with PBX1b and MEIS2b and mediates the activation of the ELA1 enhancer.

2. NKX6-1 is also a transcription factor involved in the development of pancreatic beta-cells during the secondary transition. 
Together with NKX2-2 and IRX3, controls the generation of motor neurons in the neural tube and belongs to the neural progenitor 
factors induced by Sonic Hedgehog (SHH) signals.

3.GCG and GLP1, respectively glucagon and glucagon-like peptide 1, are involved in glucose metabolism and homeostasis. 
GCG raises blood glucose levels by promoting gluconeogenesis and is the counter regulatory hormone of Insulin. 
GLP1 is a potent stimulator of Glucose-Induced Insulin Secretion (GSIS). Plays roles in gastric motility and suppresses blood glucagon levels. 
Promotes growth of the intestinal epithelium and pancreatic islet mass both by islet neogenesis and islet cell proliferation.

4. SLC2A2, also known as GLUT2, is a facilitative hexose transporter. In hepatocytes, it mediates bi-directional transport of glucose accross the plasma membranes, 
while in the pancreatic beta-cell, it is the main transporter responsible for glucose uptake and part of the cell's glucose-sensing mechanism. 
It is involved in glucose transport in the small intestine and kidney too.

To summarize, the genes in this set are involved in the specification, differentiation, growth and functionality of the pancreas, 
with a particular emphasis on the pancreatic beta-cell. Particularly, the architecture of the pancreatic islet ensures proper glucose sensing 
and homeostasis via a number of different hormones and receptors that can elicit both synergistic and antagonistic effects in the pancreas itself and other peripheral tissues.
    """

    if direct == True:
        prompt_text = direct_instructions
        prompt_text += format_placeholder
    elif customized_prompt:
        prompt_text = customized_prompt
        prompt_text += format_placeholder
    else:
        prompt_text = task_instructions
        prompt_text += score_instructions
        prompt_text += format_placeholder
        prompt_text += general_analysis_instructions
        prompt_text += example_analysis

    prompt_text += "\n\nHere are the interacting proteins:\n"
    prompt_text += f'\nProteins: '
    prompt_text += ", ".join(genes) + ".\n\n"

    if feature_df:
        prompt_text += "\n\nHere are the gene features:\n"
        prompt_text = add_gene_feature_summary(prompt_text, feature_df)

    return prompt_text


def openai_chat(context, prompt, model, temperature, seed: int = None):

    try:
        import openai
    except ImportError:
        raise ImportError('Please install the required packages: openai')

    try:
        openai.api_key = os.environ['OPENAI_API_KEY']
    except:
        raise Exception(
            "API key not found. Please set the OPENAI_API_KEY environment variable.")

    backoff_time = 10  # Start backoff time at 10 second
    max_tokens = 2000

    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": context},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        seed=seed,
        temperature=temperature,
    )

    # print(response)
    # tokens_used = response["usage"]["total_tokens"]
    tokens_used = response.usage.total_tokens
    response_content = response.choices[0].message.content
    system_fingerprint = response.system_fingerprint

    return response_content, system_fingerprint


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
    if 'nan' in unique_genes:
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

    try:
        import mygene
    except ImportError:
        raise ImportError('Please install the required packages: mygene')

    mg = mygene.MyGeneInfo()
    result = mg.querymany(all_gene, scopes="symbol",
                          fields="ensembl.gene", species=species)
    gene_id_map = {}
    for gene_info in result:
        if 'ensembl' in gene_info:
            ensembl_info = gene_info['ensembl']
            ensembl_id = ensembl_info['gene'] if isinstance(
                ensembl_info, dict) else ensembl_info[0]['gene']
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

    from suds.client import Client

    url = 'https://davidbioinformatics.nih.gov/webservice/services/DAVIDWebService?wsdl'
    client = Client(url)
    client.wsdl.services[0].setlocation(
        'https://davidbioinformatics.nih.gov/webservice/services/DAVIDWebService.DAVIDWebServiceHttpSoap11Endpoint/')
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
            f_out.write(
                f'Annotation Cluster {i}\tEnrichmentScore:{enrichment_score}\n')
            headers = ['Category', 'Term', 'Count', '%', 'Pvalue', 'Genes', 'List Total',
                       'Pop Hits', 'Pop Total', 'Fold Enrichment', 'Bonferroni', 'Benjamini', 'FDR']
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
    save_folder = os.path.join(
        data_folder, file_name.replace('.csv', '_david_results/'))
    os.makedirs(save_folder, exist_ok=True)

    overlap = 3
    initialSeed = 3
    finalSeed = 3
    linkage = 0.5
    kappa = 50

    for i in range(num_lists):
        cur_gene_list = all_onmf_df.iloc[:, i].dropna()
        cur_gene_list = map_gene_ids(gene_id_map, cur_gene_list)
        save_path = os.path.join(save_folder, f'list_{all_onmf_df.columns[i]}.csv')

        input_ids = ','.join(cur_gene_list)
        id_type = 'ENSEMBL_GENE_ID'
        list_name = f'List_{i}'
        list_type = 0

        client.service.addList(input_ids, id_type, list_name, list_type)
        term_clustering_report = client.service.getTermClusterReport(
            overlap, initialSeed, finalSeed, linkage, kappa)

        print(f'Gene list {all_onmf_df.columns[i]} ', end='\t')
        # print(term_clustering_report)
        if term_clustering_report is None:
            raise Exception(
                'No results returned from DAVID, possibly wrong token name.')

        parse_david_output(term_clustering_report, save_path)


def save_to_excel(data_folder, file_name, gene_id_map_reverse, list_name, full=True):
    """
    Save the parsed DAVID output to an Excel file, with an option to include full or filtered results.

    Parameters:
        data_folder (str): The folder where the data files are located.
        file_name (str): The name of the original CSV file containing gene data.
        gene_id_map_reverse (dict): A dictionary mapping ENSEMBL IDs to gene symbols.
        list_name (list): A list of gene list names.
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

    save_folder = os.path.join(
        data_folder, file_name.replace('.csv', '_david_results/'))
    xlsx_name = file_name.replace(
        '.csv', '_david_full.xlsx') if full else file_name.replace('.csv', '_david.xlsx')

    with pd.ExcelWriter(data_folder + xlsx_name) as writer:
        for lname in list_name:
            cur_pd = pd.read_csv(save_folder + f'list_{lname}.csv', sep='\t', names=[
                                 'Category', 'Term', 'Count', '%', 'Pvalue', 'Genes', 'List Total', 'Pop Hits', 'Pop Total', 'Fold Enrichment', 'Bonferroni', 'Benjamini', 'FDR'])
            filt1 = cur_pd['Category'] == 'UP_SEQ_FEATURE'
            filt2 = cur_pd['Pvalue'].apply(filt_fun)
            if not full:
                cur_pd = cur_pd[~(filt1 | filt2)]

            cur_pd['Term'] = cur_pd['Term'].apply(lambda x: x.split('~')[- 1])
            cur_pd['Genes'] = cur_pd['Genes'].apply(map_gene_id_reverse_list)
            cur_pd['Pvalue'] = cur_pd['Pvalue'].apply(format_pvalue)
            cur_pd['%'] = cur_pd['%'].apply(format_percent)

            ind_insert = np.where(
                cur_pd['Category'].str.startswith('Annotation Cluster'))[0]
            if len(ind_insert) > 1:
                for kk, ind in enumerate(ind_insert[1:]):
                    cur_pd = pd.DataFrame(np.insert(
                        cur_pd.values, ind + kk, values=[" "] * len(cur_pd.columns), axis=0), columns=cur_pd.columns)

            sheet_name = lname
            cur_pd.to_excel(writer, sheet_name=sheet_name, index=False)
            writer.sheets[sheet_name].set_column(0, 0, 25)
            writer.sheets[sheet_name].set_column(1, 1, 50)
            writer.sheets[sheet_name].set_column(2, 2, 5)
            writer.sheets[sheet_name].set_column(3, 3, 5)
            writer.sheets[sheet_name].set_column(5, 5, 100)