import os
import pickle
import pandas as pd
from collections import defaultdict

def convert_lakebench_groundtruth(dataset):
    base_path = '/data/qiuermu/lakebench'
    dict = {'WebTableLarge' : 'webtable_union_ground_truth.csv', 'OpenDataLarge' : 'opendata_union_ground_truth.csv'}
    gt_csv_path = os.path.join(base_path, dict[dataset])
    df = pd.read_csv(gt_csv_path)
    res = defaultdict(list)
    for x, y in zip(df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()):
        res[x].append(y)
    print(len(res))
    save_path = os.path.join(base_path, f'{dataset}Benchmark.pickle')
    pickle.dump(res, open(save_path, 'wb'))
    return res

if __name__ == '__main__':
    convert_lakebench_groundtruth('OpenDataLarge')
    convert_lakebench_groundtruth('WebTableLarge')