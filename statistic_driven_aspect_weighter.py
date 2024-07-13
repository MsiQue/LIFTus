import os
import tqdm
import pickle
import inspect
import importlib
import numpy as np
import pandas as pd
from global_info import get_csv_folder
from tokenizer import getWordsCount
from nltk.corpus import words

def delEmpty(values_raw):
    non_empty_series = values_raw.dropna()
    return non_empty_series

def NLW_Coverage(col, en_words):
    string_counts, _ = getWordsCount(col)
    if len(string_counts) == 0:
        res = 0.0
    else:
        res = sum([x[0].lower() in en_words for x in string_counts]) / len(string_counts)
    return [res]

class Feature:
    def __init__(self):
        self.feat_name = []
        self.feat_dict = {}
        self.feat_value = []

    def add(self, key, value):
        self.feat_name.append(key)
        self.feat_dict[key] = value
        self.feat_value += value

    def group_add(self, module_name, col):
        module = importlib.import_module(module_name)
        for name, obj in sorted(inspect.getmembers(module), key=lambda x: x[0]):
            if inspect.isfunction(obj):
                self.add(name, obj(col))

def column_statistic(values, sample_cnt, en_words):
# features calc before delEmpty
    all_len = len(values)

# delEmpty
    values = delEmpty(values)

# features calc after delEmpty
    length = len(values)
    sample_len = min(length, sample_cnt)

    sample_values = values.sample(sample_len, replace=False)

    feat = Feature()

    feat.add('emptyRatio', [1.0 - length / all_len if all_len > 0 else 0.0])
    feat.group_add('continuous_statistics', sample_values)
    feat.group_add('discrete_statistics', sample_values)
    feat.add('NLW_Coverage', NLW_Coverage(values, en_words))

    return np.array(feat.feat_value)

def column_statistic_by_table(args):
    table_path, sample_cnt, en_words = args
    data = pd.read_csv(table_path, lineterminator='\n')
    res = {}
    for column_name in data.columns:
        # print(table_path, column_name)
        res[column_name] = column_statistic(data[column_name], sample_cnt, en_words)
    return res

def column_statistic_all(dataset, sample_cnt):
    data_path = get_csv_folder(dataset)
    save_path = f'embedding/statistic/{dataset}_statistic.pickle'
    tableDict = {}
    en_words = set(words.words())
    for fn in tqdm.tqdm(os.listdir(data_path)):
        data_file = os.path.join(data_path, fn)
        args = (data_file, sample_cnt, en_words)
        tableDict[fn] = column_statistic_by_table(args)
    pickle.dump(tableDict, open(save_path, 'wb'))

if __name__ == '__main__':
    # for n1 in ['TUS_', 'SANTOS_']:
    #     for n2 in ['small', 'large']:
    #         dataset = n1 + n2
    #         number_encoder_all(dataset, dim = 128)

    dataset = 'test'
    sample_cnt = 200
    column_statistic_all(dataset, sample_cnt)