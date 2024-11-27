import os
import torch
import faiss
import numpy as np
import torch.nn as nn
from collections import defaultdict, Counter
from SquareSearch import SquareSearch
from global_info import get_ground_truth, get_base_info
from trainprocess import inference_model

def F(model, batch):
    info, *_batch = batch
    sep_pos = len(model.input_size_left)
    attn_output, attn_weight = model.inference(_batch[:sep_pos], _batch[sep_pos:])
    return attn_output.squeeze(), attn_weight

def get_embs(dataset, model, testData):
    tableDict, cn2id, id2cn = get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')
    attn_output, attn_weight = inference_model(F, model, testData, batch_size=128)
    column_info = []
    column_dict = defaultdict(dict)
    for i, d in enumerate(testData):
        table_name, column_name = id2cn[d[0]]
        emb = attn_output[i].detach().cpu().numpy()
        column_info.append((table_name, column_name, emb))
        column_dict[table_name][column_name] = emb
    return column_info, column_dict

def solve(M, index, query_vector, k, column_info):
    D, I = index.search(np.array([query_vector]), k)
    score = k
    for i in range(k):
        M[column_info[I[0][i]][0]] += score
        if i < k - 1 and abs(D[0][i] - D[0][i + 1]) > 1e-6:
            score -= 1

def index_init(info, method):
    vectors = list(map(lambda x: x[-1].squeeze(), info))
    vectors = np.array(vectors)

    dim, measure = vectors.shape[1], faiss.METRIC_L2
    param = method
    if method == 'Flat':
        index = faiss.IndexFlatL2(dim)
    elif method == 'SquareSearch':
        index = SquareSearch()
    else:
        index = faiss.index_factory(dim, param, measure)
    index.add(vectors)
    return index

def calc(dataset, model, testData, k, method):
    gt = get_ground_truth(dataset)
    column_info, column_dict = get_embs(dataset, model, testData)
    feat_index = index_init(column_info, method)

    ave_map = 0
    ave_precision = 0
    ave_recall = 0
    cnt = 0
    precision_list = [0 for i in range(k)]
    recall_list = [0 for i in range(k)]
    for query_table, _ in gt.items():
        if query_table.startswith('t_28'):
            # fair compare with Starmie
            continue
        M = Counter()
        for cn, emb in column_dict[query_table].items():
            query_vector = column_dict[query_table][cn]
            solve(M, feat_index, query_vector, k, column_info)

        grt = set(_)
        res = set([element[0] for element in M.most_common(k)])

        map = 0
        right_cnt = 0
        for i, element in enumerate(M.most_common(k)):
            if element[0] in grt:
                right_cnt += 1
            map += right_cnt / (i + 1)
            precision_list[i] += right_cnt / (i + 1)
            recall_list[i] += right_cnt / len(_)

        right = len(grt & res)
        if len(res) == 0:
            print(query_table)
            continue

        if len(res) != k:
            print(f'Table {query_table} not match enough')

        ave_map += map / len(res)
        ave_precision += right / len(res)
        ave_recall += right / len(_)
        cnt += 1
        M[right] += 1
        if right < k:
            print(query_table, right)
    ave_map /= cnt
    ave_precision /= cnt
    ave_recall /= cnt
    # print(ave_precision, ave_recall)
    print('Precision @ k:', [_ / cnt for _ in precision_list])
    print('Recall @ k:', [_ / cnt for _ in recall_list])
    return ave_map, ave_precision, ave_recall