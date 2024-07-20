import os.path
import pickle
import pickle5
import tqdm
import pandas as pd

def get_csv_folder(dataset):
    split_suffix = '_split_2'
    if dataset.endswith(split_suffix):
        DATAFOLDER = f'positive_sampling/{dataset.split(split_suffix)[0]}'
    else:
        DATAFOLDER = f'data_lake/{dataset}'
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

def get_base_info_all(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            dataset = n1 + n2
            save_path = f'{base_path}/{dataset}_base_info.pickle'
            get_base_info(dataset, save_path)

    # _split_2
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            dataset = n1 + n2 + '_split_2'
            save_path = f'{base_path}/{dataset}_base_info.pickle'
            get_base_info(dataset, save_path)

if __name__ == '__main__':
    get_base_info_all('base_info')