from global_info import get_csv_folder
import os
import math
import tqdm
import pickle
import numpy as np
import pandas as pd

def unzip(sorted_pairs):
    res = []
    for value, cnt in sorted_pairs:
        res.extend([value] * cnt)
    return np.array(res)

def ranking_encoder_by_column(normalized_numbers, dim):
    L = len(normalized_numbers)
    if L < dim:
        return np.concatenate([np.zeros((dim - L)), normalized_numbers])
    delta = (L - 1) // (dim - 1)
    return normalized_numbers[::delta][:dim]

def histogram_encoder_by_column(sorted_pairs, dim):
    _min = sorted_pairs[0][0]
    _max = sorted_pairs[-1][0]
    delta = (_max - _min) / dim
    res = [0 for i in range(dim)]
    cur_bin = 0
    cur_R_margin = _min + delta
    for value, cnt in sorted_pairs:
        while value >= cur_R_margin:
            if cur_bin < dim - 1:
                cur_bin += 1
            cur_R_margin += delta
        res[cur_bin] += cnt
    assert sum([x[1] for x in sorted_pairs]) == sum(res)
    assert res[-1] > 0
    return np.array(res)

def sgn(x):
    eps = 1e-9
    if x > eps:
        return 1
    elif x < -eps:
        return -1
    else:
        return 0

def f_base(a, x):
    return sgn(x) * (abs(x) ** a)

def f_macro(a, x):
    return sgn(x) * math.log(abs(x) + 1) / math.log(a)

def f_micro(a, x):
    return math.sin(a*x)

def magnitude_encoder_by_value(x, dim):
    res = []
    w = [lambda x: x / dim, lambda x: x + 1, lambda x: x]
    for i, f in enumerate([f_base, f_macro, f_micro]):
        res.append(np.array([f(w[i](a), x) for a in range(1, dim + 1)]))
    return np.vstack(res)

def number_encoder_by_column(number_pairs, dim):
    if len(number_pairs) == 0:
        return np.zeros((14, dim))
    sorted_pairs = sorted(number_pairs, key=lambda x: x[0])
    sorted_numbers = np.sort(unzip(sorted_pairs))
    _min = sorted_numbers[0]
    _max = sorted_numbers[-1]
    _range = max(_max - _min, 0)
    _ave = sum(sorted_numbers) / len(sorted_numbers)
    if _max - _min < 1e-9:
        # All Same
        ranking_emb = np.ones(dim)
    else:
        normalized_numbers = (sorted_numbers - _min) / _range
        ranking_emb = ranking_encoder_by_column(normalized_numbers, dim)
    return np.vstack((ranking_emb.reshape(1, -1),
                      histogram_encoder_by_column(sorted_pairs, dim).reshape(1, -1),
                      magnitude_encoder_by_value(_min, dim),
                      magnitude_encoder_by_value(_max, dim),
                      magnitude_encoder_by_value(_range, dim),
                      magnitude_encoder_by_value(_ave, dim),
                      ))

def number_encoder_by_table(args):
    table_path, number_topK, dim = args
    data = pd.read_csv(table_path, lineterminator='\n')
    res = {}
    for column_name in data.columns:
        res[column_name] = number_encoder_by_column(number_topK[column_name], dim)
    return res

def number_encoder_all(dataset, dim, string_k = 64, number_k = 512):
    data_path = get_csv_folder(dataset)
    save_path = f'embedding/number/{dataset}_number_emb_{dim}_dim.pickle'
    tokenized_path = f'step_result/tokens/{dataset}_tokens_{string_k}_string_{number_k}_number.pickle'
    tokenized_info = pickle.load(open(tokenized_path, 'rb'))
    tableDict = {}
    for fn in tqdm.tqdm(os.listdir(data_path)):
        data_file = os.path.join(data_path, fn)
        args = (data_file, tokenized_info[fn][1], dim)
        tableDict[fn] = number_encoder_by_table(args)
    pickle.dump(tableDict, open(save_path, 'wb'))

if __name__ == '__main__':
    # for n1 in ['TUS_', 'SANTOS_']:
    #     for n2 in ['small', 'large']:
    #         dataset = n1 + n2
    #         number_encoder_all(dataset, dim = 128)

    dataset = 'test'
    number_encoder_all(dataset, dim = 128)