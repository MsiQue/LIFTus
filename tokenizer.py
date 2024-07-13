import tqdm
import os
import pickle
import pandas as pd
from collections import defaultdict
from parallel import solve
from global_info import get_csv_folder
from filter import filter

def getWordsCount(col):
    counts = col.value_counts().to_dict()
    string_counts = defaultdict(int)
    number_counts = defaultdict(int)
    for k, v in counts.items():
        List_string, List_number = filter(str(k))
        for x in List_string:
            string_counts[x] += v
        for x in List_number:
            number_counts[x] += v
    string_counts = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)
    number_counts = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
    return string_counts, number_counts
def tokenize_column_by_table(args):
    table_path, string_k, number_k = args
    data = pd.read_csv(table_path, lineterminator='\n')
    string_topK = {}
    number_topK = {}
    for column_name in data.columns:
        string_counts, number_counts = getWordsCount(data[column_name])
        string_topK[column_name] = set(string_counts[:string_k])
        number_topK[column_name] = set(number_counts[:number_k])
    return (string_topK, number_topK)

def tokenizer_all(data_path, save_path, string_k, number_k):
    tableDict = {}
    for fn in tqdm.tqdm(os.listdir(data_path)):
        data_file = os.path.join(data_path, fn)
        args = (data_file, string_k, number_k)
        tableDict[fn] = tokenize_column_by_table(args)
    pickle.dump(tableDict, open(save_path, 'wb'))

def tokenizer_all_parallel(data_path, save_path, string_k, number_k, sequential_cnt, parallel_cnt):
    message = f'get {save_path}'
    args_dict = {}
    for table_name in tqdm.tqdm(os.listdir(data_path)):
        args_dict[table_name] = (os.path.join(data_path, table_name), string_k, number_k)
    solve(message, sequential_cnt, parallel_cnt, tokenize_column_by_table, args_dict, os.listdir(data_path), save_path)

if __name__ == '__main__':
    string_k = 64
    number_k = 512
    # for n1 in ['TUS_', 'SANTOS_']:
    #     for n2 in ['small', 'large']:
    #         dataset = n1 + n2
    #         data_path = get_csv_folder(dataset)
    #         save_path = f'step_result/tokens/{dataset}_tokens_{string_k}_string_{number_k}_number.pickle'
    #         tokenizer_all(data_path, save_path, string_k, number_k)
    #         tokenizer_all_parallel(data_path, save_path, string_k, number_k, 50, 5)

    dataset = 'test'
    data_path = get_csv_folder(dataset)
    save_path = f'step_result/tokens/{dataset}_tokens_{string_k}_string_{number_k}_number.pickle'
    tokenizer_all(data_path, save_path, string_k, number_k)
    # tokenizer_all_parallel(data_path, save_path, string_k, number_k, 50, 5)