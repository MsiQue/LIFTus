import numpy as np
from utils import __
from parallel import solve
from global_info import get_csv_folder
import os
import time
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

def convertFunc_bert(tokenizer, model, device, text):
    # t1 = time.time()
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    word_embeddings = last_hidden_state.mean(dim=1)
    # print(time.time() - t1)
    return word_embeddings

# def text2vec(M, text):
#     t1 = time.time()
#     tokenizer = BertTokenizer.from_pretrained('/data/qiuermu/bert_v2/bert_pretrain')
#     print(time.time() - t1)
#     model = BertModel.from_pretrained('/data/qiuermu/bert_v2/bert_pretrain')
#     print(time.time() - t1)
#     inputs = tokenizer(text, return_tensors="pt")
#     print(time.time() - t1)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     print(time.time() - t1)
#     last_hidden_state = outputs.last_hidden_state
#     word_embeddings = last_hidden_state.mean(dim=1)
#     return word_embeddings

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
        res.append((s, emb))
    return res

def get_table_paragraph_info_by_column(args):
    table_path, sample_cnt, method, kwargs = args
    df = pd.read_csv(table_path, engine='python')
    res = {}
    for column_name in df.columns:
        res[__(column_name)] = col_paragraph2vectors(df[column_name], sample_cnt, method, **kwargs)
    return res

def get_paragraph_info(sample_cnt, method, parallel_cnt = 30, sequential_limit = 200, **kwargs):
    start_time = time.time()
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            for n3 in ['', '_split_2']:
                dataset = n1 + n2 + n3
                csv_folder = get_csv_folder(dataset)
                N = len(os.listdir(csv_folder))
                sequential_cnt = min(N // parallel_cnt + 1, sequential_limit)
                # get_info_parallel_step(csv_folder, sequential_cnt, parallel_cnt, dim, output_dict_file=f'vectors/arrayEmb/{dataset}_dim_{dim}.pickle')

                output_dict_file = f'vectors/paragraphEmb/{dataset}_sample_{sample_cnt}_paragraph__{method}.pickle'
                message = f'paragraph info:[ {csv_folder}__{output_dict_file} ]'
                todo_list = os.listdir(csv_folder)
                args_dict = {}
                for table_name in todo_list:
                    args_dict[table_name] = (os.path.join(csv_folder, table_name), sample_cnt, method, kwargs)
                solve(message, sequential_cnt, parallel_cnt, get_table_paragraph_info_by_column, args_dict, todo_list, output_dict_file)

    print(f'get num info cost {time.time() - start_time}')

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained('/data/qiuermu/bert_v2/bert_pretrain')
    model = BertModel.from_pretrained('/data/qiuermu/bert_v2/bert_pretrain').to(device)
    get_paragraph_info(sample_cnt = 32, method = 'bert', parallel_cnt = 1, sequential_limit = 30,  convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)
