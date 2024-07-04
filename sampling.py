from table import *
from utils import load_vectors, getColPairVectors, _
import random

def genPositive_all(dataset, table_info, table_info_split_2, topk, fasttext_info, sample_cnt):
    pos_sample_save_path = 'pos_neg_sample/pos_sample_all_' + dataset + '.pth'
    if os.path.exists(pos_sample_save_path):
        return pickle.load(open(pos_sample_save_path, 'rb'))

    tfidf_topk_save_path = f'tfidf/{dataset}_pos_sample_top100.pickle'
    col1_list = []
    col2_list = []
    for table_name, table in tqdm.tqdm(table_info.items()):
        if table.row_len < 2:
            continue
        for column_name in table.columnDict.keys():
            for i in range(1, sample_cnt + 1):
                for j in range(i + 1, sample_cnt + 1):
                    try:
                        col1_list.append(_(table_info_split_2[table_name.split('.')[0] + f'____{i}.csv'].columnDict, column_name))
                        col2_list.append(_(table_info_split_2[table_name.split('.')[0] + f'____{j}.csv'].columnDict, column_name))
                    except:
                        print(table_name, i, j, '='*60)

    res = getColPairVectors(tfidf_topk_save_path, topk, fasttext_info, col1_list, col2_list, 1)
    pickle.dump(res, open(pos_sample_save_path, 'wb'))
    return res

def genNegative_step(table_info):
    L = list(table_info.keys())
    while True:
        x = random.choice(L)
        y = random.choice(L)
        # 纯随机选，不强制表名前缀不同，只要求表不同
        if x == y:
            continue
        a = random.choice(list(table_info[x].columnDict.keys()))
        b = random.choice(list(table_info[y].columnDict.keys()))
        return table_info[x].columnDict[a], table_info[y].columnDict[b]

def genNegative(dataset, cnt, table_info, topk, fasttext_info):
    neg_sample_save_path = 'pos_neg_sample/neg_sample_' + dataset + '_' + str(cnt) + '.pth'
    if os.path.exists(neg_sample_save_path):
        return pickle.load(open(neg_sample_save_path, 'rb'))

    tfidf_topk_save_path = f'tfidf/{dataset}_top100.pickle'
    col1_list = []
    col2_list = []
    for i in tqdm.tqdm(range(cnt)):
        x, y = genNegative_step(table_info)
        col1_list.append(x)
        col2_list.append(y)
    res = getColPairVectors(tfidf_topk_save_path, topk, fasttext_info, col1_list, col2_list, 0)
    pickle.dump(res, open(neg_sample_save_path, 'wb'))
    return res

if __name__ == '__main__':
    # dataset = 'TUS_small'
    # dataset = 'TUS_small'
    dataset = 'SANTOS_large'
    # dataset = 'TUS_large'
    save_path = 'table_info/' + dataset + '_table_info.pickle'
    table_info = pickle.load(open(save_path, 'rb'))
    table_info_split_2 = pickle.load(open(f'table_info_split_2_times_3/{dataset}_table_info_split_2_times_3.pickle', 'rb'))
    fasttext_info = load_vectors('/data/qiuermu/fasttext/wiki-news-300d-1M.vec')

    if dataset == 'SANTOS_small':
        neg_sample_cnt = 100000
    elif dataset == 'TUS_small':
        neg_sample_cnt = 250000
    elif dataset == 'TUS_large':
        neg_sample_cnt = 850000
    elif dataset == 'SANTOS_large':
        neg_sample_cnt = 1800000

    # genPositive_all(dataset, table_info, table_info_split_2, 10, fasttext_info, 6)
    genNegative(dataset, neg_sample_cnt, table_info, 10, fasttext_info)
