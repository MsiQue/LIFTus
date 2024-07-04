import os
import pickle
import pickle5
import tqdm
import pandas as pd

def get_csv_folder(dataset):
    if os.name == 'nt':
        if dataset == 'TUS_small':
            DATAFOLDER = 'D:\Dataset\LIFTus-dataset\TUS_small\datalake'
        elif dataset == 'TUS_large':
            DATAFOLDER = 'D:\Dataset\LIFTus-dataset\TUS_large\datalake'
        elif dataset == 'SANTOS_small':
            DATAFOLDER = 'D:\Dataset\LIFTus-dataset\labeled_benchmark\datalake'
        elif dataset == 'SANTOS_large':
            DATAFOLDER = 'D:\Dataset\LIFTus-dataset\\real_tables_benchmark\datalake'
        elif dataset.endswith('_split_2'):
            DATAFOLDER = f'D:\Dataset\LIFTus-dataset\{dataset.split("_split_2")[0]}'
        else:
            DATAFOLDER = ''
    else:
        if dataset == 'TUS_small':
            DATAFOLDER = '/data/qiuermu/starmie/data/table-union-search-benchmark/small/benchmark'
        elif dataset == 'TUS_large':
            DATAFOLDER = '/data/qiuermu/dataset/TUS_Large/tables'
        elif dataset == 'SANTOS_small':
            DATAFOLDER = '/data/qiuermu/santos/benchmark/santos_benchmark/dataLake'
        elif dataset == 'SANTOS_large':
            DATAFOLDER = '/data/qiuermu/santos/benchmark/real_tables_benchmark/dataLake'
        elif dataset.endswith('_split_2'):
            DATAFOLDER = f'/data/qiuermu/test_74/positive_sampling/{dataset.split("_split_2")[0]}'
        else:
            DATAFOLDER = ''

    return DATAFOLDER

def get_ground_truth(dataset):
    if 'TUS_small' in dataset:
        gt_path = 'ground_truth/TUS_smallUnionBenchmark.pickle'
    elif 'TUS_large' in dataset:
        gt_path = 'ground_truth/TUS_largeUnionBenchmark.pickle'
    elif 'SANTOS_small' in dataset:
        gt_path = 'ground_truth/santosUnionBenchmark.pickle'
    elif 'SANTOS_large' in dataset:
        gt_path = 'ground_truth/real_tablesUnionBenchmark_3.pickle'
    return pickle5.load(open(gt_path, 'rb'))

def get_base_info(dataset, save_path):

    if os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    csv_folder = get_csv_folder(dataset)
    tableDict = {}
    id2cn = []
    cn2id = {}
    for tn in tqdm.tqdm(os.listdir(csv_folder)):
        table_path = os.path.join(csv_folder, tn)
        tableDict[tn] = [(s.rstrip('\r') if s.endswith('\r') else s) for s in list(pd.read_csv(table_path, engine='python'))]
        for cn in tableDict[tn]:
            cn2id[(tn, cn)] = len(id2cn)
            id2cn.append((tn, cn))
            if '\r' in cn:
                print(f'exist \\r in {dataset} {tn} {cn}')

    pickle.dump((tableDict, cn2id, id2cn), open(save_path, 'wb'))


if __name__ == '__main__':
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            dataset = n1 + n2
            save_path = f'base_info/{dataset}_base_info.pickle'
            get_base_info(dataset, save_path)

    # _split_2
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            dataset = n1 + n2 + '_split_2'
            save_path = f'base_info/{dataset}_base_info.pickle'
            get_base_info(dataset, save_path)