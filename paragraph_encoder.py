import numpy as np
from utils import __
from parallel import solve
from global_info import get_csv_folder
import os
import time
import tqdm
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pickle

def convertFunc_bert(tokenizer, model, device, text):
    # t1 = time.time()
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    word_embeddings = last_hidden_state.mean(dim=1)
    # print(time.time() - t1)
    return word_embeddings

def col_paragraph2vectors(_col, sample_cnt, method, **kwargs):
    col = _col.copy()
    col = col.dropna()
    col = col.replace({True: 1, False: 0})
    col = col.astype(str)
    lengths = col.str.len()
    col = col.iloc[lengths.argsort()[::-1]]
    col = col.drop_duplicates()
    res = []
    for i in range(len(col)):
        if i == sample_cnt:
            break
        s = col.iloc[i]
        if method == 'bert':
            emb = kwargs['convertFunc'](kwargs['tokenizer'], kwargs['model'], kwargs['device'], s)
        res.append((s, emb.squeeze().detach().cpu().numpy()))
    return res

def get_table_paragraph_info_by_column(args):
    table_path, sample_cnt, method, kwargs = args
    df = pd.read_csv(table_path, engine='python')
    res = {}
    for column_name in df.columns:
        res[__(column_name)] = col_paragraph2vectors(df[column_name], sample_cnt, method, **kwargs)
    return res

def paragraph_encoder_all(dataset, sample_cnt, method, is_parallel = False, **kwargs):
    csv_folder = get_csv_folder(dataset)
    N = len(os.listdir(csv_folder))
    output_dict_file_root = 'embeddings/paragraph'
    if not os.path.exists(output_dict_file_root):
        os.makedirs(output_dict_file_root)
    output_dict_file = os.path.join(output_dict_file_root, f'{dataset}_sample_{sample_cnt}_paragraph__{method}.pickle')

    if os.path.exists(output_dict_file):
        print('Complete paragraph_encoder_all !')
        return

    todo_list = os.listdir(csv_folder)
    args_dict = {}
    for table_name in todo_list:
        args_dict[table_name] = (os.path.join(csv_folder, table_name), sample_cnt, method, kwargs)

    if is_parallel:
        parallel_cnt = 1 #kwargs['parallel_cnt']
        sequential_limit = kwargs['sequential_limit']
        sequential_cnt = min(N // parallel_cnt + 1, sequential_limit)

        message = f'paragraph info:[ {csv_folder}__{output_dict_file} ]'
        solve(message, sequential_cnt, parallel_cnt, get_table_paragraph_info_by_column, args_dict, todo_list, output_dict_file)
    else:
        res = {}
        for table_name in tqdm.tqdm(todo_list):
            res[table_name] = get_table_paragraph_info_by_column(args_dict[table_name])
        pickle.dump(res, open(output_dict_file, 'wb'))

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    LM_path = 'LM/bert'
    tokenizer = BertTokenizer.from_pretrained(LM_path)
    model = BertModel.from_pretrained(LM_path).to(device)

    print('finishing loading LM')

    # paragraph_encoder_all('test', 128, 'bert', is_parallel=False, convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)
    # paragraph_encoder_all('test_split_2', 128, 'bert', is_parallel=False, convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)

    paragraph_encoder_all('SANTOS_small', 128, 'bert', is_parallel=False, convertFunc=convertFunc_bert, tokenizer=tokenizer,
                          model=model, device=device)
    paragraph_encoder_all('SANTOS_small_split_2', 128, 'bert', is_parallel=False, convertFunc=convertFunc_bert,
                          tokenizer=tokenizer,
                          model=model, device=device)

    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            for n3 in ['', '_split_2']:
                dataset = n1 + n2 + n3
                paragraph_encoder_all(dataset, 128, 'bert', is_parallel=True, parallel_cnt = 1, sequential_limit = 30, convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)

    # get_paragraph_info(sample_cnt = 32, method = 'bert', parallel_cnt = 1, sequential_limit = 30,  convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)