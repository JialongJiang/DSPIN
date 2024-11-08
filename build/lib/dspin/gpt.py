import openai
import csv
import pandas as pd
from typing import List

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
