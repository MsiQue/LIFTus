import numpy as np
from utils import __
from parallel import solve
from global_info import get_csv_folder, get_base_info
import os
import time
import tqdm
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pickle
import copy
from torch.utils.data import DataLoader, TensorDataset
from trainprocess import inference_model
from collections import defaultdict

def convertFunc_bert(tokenizer, model, device, text):
    # t1 = time.time()
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    word_embeddings = last_hidden_state.mean(dim=1)
    # print(time.time() - t1)
    return word_embeddings

def tokenizerFunc(tokenizer, text):
    return tokenizer(text, max_length=32, padding='max_length', truncation=True, return_tensors="pt")

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
        if 'bert' in method:
            if 'tokenizer' in method:
                emb = kwargs['tokenizerFunc'](kwargs['tokenizer'], s)
                res.append((s, emb['input_ids'].squeeze(), emb['token_type_ids'].squeeze(), emb['attention_mask'].squeeze()))
            else:
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

def getTokenizerResult(dataset, sample_cnt, method, kwargs):
    token_output_dict_file_root = 'step_result/paragraph_tokenizer_result'
    if not os.path.exists(token_output_dict_file_root):
        os.makedirs(token_output_dict_file_root)
    token_output_dict_file = os.path.join(token_output_dict_file_root, f'{dataset}_sample_{sample_cnt}_tokens__{method}.pickle')

    if os.path.exists(token_output_dict_file):
        print(f'Complete paragraph tokenizer {dataset} !')
        return

    csv_folder = get_csv_folder(dataset)
    todo_list = os.listdir(csv_folder)
    _kwargs = copy.deepcopy(kwargs)
    _kwargs.pop('model')
    _kwargs.pop('device')
    args_dict = {}
    for table_name in todo_list:
        args_dict[table_name] = (os.path.join(csv_folder, table_name), sample_cnt, method + 'tokenizer', _kwargs)

    N = len(os.listdir(csv_folder))
    parallel_cnt = kwargs['parallel_cnt']
    sequential_limit = kwargs['sequential_limit']
    sequential_cnt = min(N // parallel_cnt + 1, sequential_limit)

    message = f'tokenizer : [ {csv_folder}__{token_output_dict_file} ]'
    solve(message, sequential_cnt, parallel_cnt, get_table_paragraph_info_by_column, args_dict, todo_list, token_output_dict_file)

def paragraph_encoder_batch(dataset, sample_cnt, method, kwargs):
    token_output_dict_file_root = 'step_result/paragraph_tokenizer_result'
    token_output_dict_file = os.path.join(token_output_dict_file_root, f'{dataset}_sample_{sample_cnt}_tokens__{method}.pickle')
    tokenizerResult = pickle.load(open(token_output_dict_file, 'rb'))

    tableDict, cn2id, id2cn = get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')

    cn_id_list = []
    input_ids_list = []
    attention_mask_list = []
    for table_name, table in tqdm.tqdm(tokenizerResult.items()):
        for column_name, column_paragraphs in table.items():
            if (table_name, column_name) not in cn2id:
                print(table_name, column_name)
                continue
            for i, paragraph in enumerate(column_paragraphs):
                cn_id_list.append(cn2id[table_name, column_name])
                input_ids_list.append(paragraph[1])
                attention_mask_list.append(paragraph[3])

    device = kwargs['device']
    cn_id_list = torch.tensor(cn_id_list, dtype=torch.long, device=device).unsqueeze(1)
    input_ids_list = torch.stack(input_ids_list).to(device)
    attention_mask_list = torch.stack(attention_mask_list).to(device)

    Data = TensorDataset(cn_id_list, input_ids_list, attention_mask_list)
    F = lambda model, batch : (batch[0], model(input_ids = batch[1], attention_mask = batch[2]).last_hidden_state.mean(dim=1))
    cn_list, vec_list = inference_model(F, model = kwargs['model'], Data=Data, batch_size=128)

    result_dict_temp = defaultdict(list)
    # for k, v in zip(*res):
    #     result_dict_temp[k.item()].append(v)

    for k, v in zip(cn_list, vec_list):
        # print(k.item(), len(result_dict_temp), len(result_dict_temp[k.item()]))
        result_dict_temp[k.item()].append([v.detach().cpu().numpy()])

    result_dict = defaultdict(dict)
    for k, v in result_dict_temp.items():
        table_name, column_name = id2cn[k]
        result_dict[table_name][column_name] = v
    return result_dict

def paragraph_encoder_all(dataset, sample_cnt, method, is_parallel = False, **kwargs):
    output_dict_file_root = 'embeddings/paragraph'
    if not os.path.exists(output_dict_file_root):
        os.makedirs(output_dict_file_root)
    output_dict_file = os.path.join(output_dict_file_root, f'{dataset}_sample_{sample_cnt}_paragraph__{method}.pickle')

    if os.path.exists(output_dict_file):
        print(f'Already complete paragraph_encoder_all on {output_dict_file} before!')
        return

    start_time = time.time()
    print(f'Start paragraph_encoder_all on {dataset}.')

    if is_parallel:
        getTokenizerResult(dataset, sample_cnt, method, kwargs)
        res = paragraph_encoder_batch(dataset, sample_cnt, method, kwargs)
    else:
        csv_folder = get_csv_folder(dataset)
        args_dict = {}
        todo_list = os.listdir(csv_folder)
        for table_name in todo_list:
            args_dict[table_name] = (os.path.join(csv_folder, table_name), sample_cnt, method, kwargs)
        res = {}
        for table_name in todo_list:
            res[table_name] = get_table_paragraph_info_by_column(args_dict[table_name])
    pickle.dump(res, open(output_dict_file, 'wb'))

    print(f'Complete paragraph_encoder_all on {dataset} using {time.time() - start_time} s.')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    LM_path = 'LM/bert'
    tokenizer = BertTokenizer.from_pretrained(LM_path)
    model = BertModel.from_pretrained(LM_path).to(device)
    model.eval()

    print('finishing loading LM')

    t1 = time.time()
    # paragraph_encoder_all('test', 32, 'bert', is_parallel=True, parallel_cnt = 30, sequential_limit = 30, tokenizerFunc = tokenizerFunc, tokenizer = tokenizer, model = model, device = device)
    paragraph_encoder_all('SANTOS_small', 32, 'bert', is_parallel=True, parallel_cnt = 30, sequential_limit = 30, tokenizerFunc = tokenizerFunc, tokenizer = tokenizer, model = model, device = device)
    paragraph_encoder_all('SANTOS_small_split_2', 32, 'bert', is_parallel=True, parallel_cnt = 30, sequential_limit = 30, tokenizerFunc = tokenizerFunc, tokenizer = tokenizer, model = model, device = device)
    print(time.time() - t1)

    # t1 = time.time()
    # paragraph_encoder_all('test', 32, 'bert', is_parallel=False, convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)
    # paragraph_encoder_all('test_split_2', 32, 'bert', is_parallel=False, convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)
    # print(time.time() - t1)

    # paragraph_encoder_all('SANTOS_small', 32, 'bert', is_parallel=False, convertFunc=convertFunc_bert, tokenizer=tokenizer,
    #                       model=model, device=device)
    # paragraph_encoder_all('SANTOS_small_split_2', 32, 'bert', is_parallel=False, convertFunc=convertFunc_bert,
    #                       tokenizer=tokenizer,
    #                       model=model, device=device)
    #
    # for n1 in ['TUS_', 'SANTOS_']:
    #     for n2 in ['small', 'large']:
    #         for n3 in ['', '_split_2']:
    #             dataset = n1 + n2 + n3
    #             paragraph_encoder_all(dataset, 32, 'bert', is_parallel=True, parallel_cnt = 1, sequential_limit = 30, convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)

    # get_paragraph_info(sample_cnt = 32, method = 'bert', parallel_cnt = 1, sequential_limit = 30,  convertFunc = convertFunc_bert, tokenizer = tokenizer, model = model, device = device)