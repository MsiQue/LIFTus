import random
import pickle
from global_info import get_csv_folder
from parallel import solve
from collections import defaultdict
import tqdm
import time
import os
import pandas as pd
def merge(s1, s2):
    res = ''
    for x, y in zip(s1, s2):
        if x == '*' or y == '*':
            res += '*'
        elif x != y:
            res += '*'
        else:
            res += x
    return res

def getPattern(v):
    values = v.reset_index(drop=True)
    Len = len(values)
    if Len == 0:
        return None
    # if '*' in str(values[0]):
    #     print(col.belong_to, col.name, 0)
    pattern = str(values[0])
    for i in range(1, Len):
        # if '*' in str(values[i]):
        #     print(col.belong_to, col.name, i)
        pattern = merge(pattern, str(values[i]))
        if pattern == '*' * len(str(values[i])):
            return None
    return pattern

def getColPatterns_step(col, n_sample, n_check, needReverse=False):
    res = []
    n_sample = min(n_sample, len(col))
    sample_idx = random.sample(range(len(col)), n_sample)
    for idx in sample_idx:
        end_pos = min(idx + n_check, len(col))
        p = getPattern(col[idx:end_pos])
        if p is None:
            res.append(col[idx])
        else:
            res.append(p)
    if needReverse == True:
        return [s[::-1] for s in res]
    else:
        return res

def getColPatterns(col, n_sample, n_check):
    str_col = col.astype(str)
    clean_col = str_col.dropna()
    sorted_col_1 = clean_col.sort_values().reset_index(drop=True)
    sorted_col_2 = clean_col.apply(lambda x: x[::-1]).sort_values().reset_index(drop=True)
    return getColPatterns_step(sorted_col_1, n_sample, n_check) + getColPatterns_step(sorted_col_2, n_sample, n_check, needReverse=True)

def getColPatterns_by_table(args):
    table_path, n_sample, n_check = args
    data = pd.read_csv(table_path, lineterminator='\n')
    res = defaultdict(list)
    for column_name, column_type_from_pandas in zip(data.columns, data.dtypes):
        res[column_name] = getColPatterns(data[column_name], n_sample, n_check)
    return res

def getPattern_all(data_path, save_path, n_sample, n_check):
    tableDict = {}
    for fn in os.listdir(data_path):
        data_file = os.path.join(data_path, fn)
        args = (data_file, n_sample, n_check)
        tableDict[fn] = getColPatterns_by_table(args)
    pickle.dump(tableDict, open(save_path, 'wb'))

def getPattern_all_parallel(data_path, save_path, n_sample, n_check, sequential_cnt, parallel_cnt):
    message = f'get {save_path}'
    args_dict = {}
    for table_name in os.listdir(data_path):
        args_dict[table_name] = (os.path.join(data_path, table_name), n_sample, n_check)
    solve(message, sequential_cnt, parallel_cnt, getColPatterns_by_table, args_dict, os.listdir(data_path), save_path)

def summarize_Pattern(dataset, n_sample, n_check, is_parallel = False, sequential_cnt=10, parallel_cnt=30):
    path = get_csv_folder(dataset)
    save_path_root = 'step_result/pattern'
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    save_path = os.path.join(save_path_root, f'{dataset}_pattern_{n_sample}_{n_check}.pickle')

    if os.path.exists(save_path):
        print(f'Already complete summarize_Pattern on {save_path} before!')
        return

    start_time = time.time()
    print(f'Start summarize_Pattern on {dataset}.')
    if is_parallel:
        getPattern_all_parallel(path, save_path, n_sample, n_check, sequential_cnt, parallel_cnt)
    else:
        getPattern_all(path, save_path, n_sample, n_check)
    print(f'Complete summarize_Pattern on {dataset} using {time.time() - start_time} s.')

if __name__ == '__main__':
    summarize_Pattern('test' + '_split_2', 10, 10)
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            summarize_Pattern(n1 + n2 + '_split_2', 10, 10)
    # summarize_Pattern('SANTOS_small', 10, 10)
    # summarize_Pattern('SANTOS_large', 10, 10)