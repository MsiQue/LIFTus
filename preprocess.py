from torch.utils.data import TensorDataset
from global_info import get_base_info
from utils import _
import numpy as np
import pickle
import torch
import os

# def getData_step(col_list, emb_source_list, device):
#     emb = []
#     for source in emb_source_list:
#         if isinstance(source, str) and os.path.isfile(source):
#             emb_dict = pickle.load(open(source, 'rb'))
#             dim = list(list(emb_dict.values())[0].values())[0].shape[0]
#             res = []
#             for table_name, column_name in col_list:
#                 if (table_name in emb_dict) and (column_name in emb_dict[table_name]):
#                     res.append(emb_dict[table_name][column_name])
#                 else:
#                     res.append(np.zeros(dim))
#             emb.append(torch.tensor(res, dtype=torch.float32).unsqueeze(dim=1).to(device))
#         elif isinstance(source, list):
#             emb_dict = pickle.load(open(source[0], 'rb'))
#             k = source[1]
#             dim = source[2]
#             res = []
#             for table_name, column_name in col_list:
#                 t = _(emb_dict[table_name], column_name)
#                 if len(t) > 0:
#                     r = np.array([x[-1] for x in t[:k]])
#                     if len(r) < k:
#                         r = np.concatenate([r, np.zeros((k - len(r), dim))], axis=0)
#                 else:
#                     r = np.zeros((k, dim))
#                 res.append(r)
#             emb.append(torch.tensor(res, dtype=torch.float32).to(device))
#         else:
#             pass
#     return emb
#
# def getData(cn2id, col_list, left_list, right_list, device):
#     info = torch.tensor([cn2id[table_name, column_name] for table_name, column_name in col_list]).to(device)
#     left_emb = getData_step(col_list, left_list, device)
#     right_emb = getData_step(col_list, right_list, device)
#     return info, left_emb, right_emb
#
# def getTrainData(dataset, left_list, right_list, device):
#     tableDict, cn2id, id2cn = get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')
#     col_list_1 = []
#     col_list_2 = []
#     for table_name, columnList in tableDict.items():
#         if table_name.endswith('____2.csv'):
#             continue
#         table_name_2 = table_name.split('____1.csv')[0] + '____2.csv'
#         for column_name in columnList:
#             if column_name not in tableDict[table_name_2]:
#                 print(f'{column_name} not in table {table_name_2}')
#                 continue
#             col_list_1.append((table_name, column_name))
#             col_list_2.append((table_name_2, column_name))
#
#     info_1, left_list_1, right_list_1 = getData(cn2id, col_list_1, left_list, right_list, device)
#     info_2, left_list_2, right_list_2 = getData(cn2id, col_list_2, left_list, right_list, device)
#     return TensorDataset(info_1, *left_list_1, *right_list_1, info_2, *left_list_2, *right_list_2)
#
# def getTestData(dataset, left_list, right_list, device):
#     tableDict, cn2id, id2cn = get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')
#     col_list = []
#     for table_name, columnList in tableDict.items():
#         for column_name in columnList:
#             col_list.append((table_name, column_name))
#
#     info, left_list, right_list = getData(cn2id, col_list, left_list, right_list, device)
#     return TensorDataset(info, *left_list, *right_list)

def getData_step(col_list, emb_source_list):
    emb = []
    for source in emb_source_list:
        if isinstance(source, str) and os.path.isfile(source):
            emb_dict = pickle.load(open(source, 'rb'))
            dim = list(list(emb_dict.values())[0].values())[0].shape[0]
            res = []
            for table_name, column_name in col_list:
                if (table_name in emb_dict) and (column_name in emb_dict[table_name]):
                    res.append(emb_dict[table_name][column_name])
                else:
                    res.append(np.zeros(dim))
            emb.append(torch.tensor(res, dtype=torch.float32).unsqueeze(dim=1))
        elif isinstance(source, list):
            emb_dict = pickle.load(open(source[0], 'rb'))
            k = source[1]
            dim = source[2]

            res = []
            for table_name, column_name in col_list:
                t = _(emb_dict[table_name], column_name)
                if (t is not None) and len(t) > 0:
                    r = np.array([x[-1] for x in t[:k]])
                    if len(r) < k:
                        r = np.concatenate([r, np.zeros((k - len(r), dim))], axis=0)
                else:
                    r = np.zeros((k, dim))
                res.append(r)
            emb.append(torch.tensor(res, dtype=torch.float32))
        else:
            pass
    return emb

def getData(cn2id, col_list, left_list, right_list):
    info = torch.tensor([cn2id[table_name, column_name] for table_name, column_name in col_list])
    left_emb = getData_step(col_list, left_list)
    right_emb = getData_step(col_list, right_list)
    return info, left_emb, right_emb

def getTrainData(dataset, left_list, right_list):
    tableDict, cn2id, id2cn = get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')
    col_list_1 = []
    col_list_2 = []
    for table_name, columnList in tableDict.items():
        if table_name.endswith('____2.csv'):
            continue
        table_name_2 = table_name.split('____1.csv')[0] + '____2.csv'
        for column_name in columnList:
            if column_name not in tableDict[table_name_2]:
                print(f'{column_name} not in table {table_name_2}')
                continue
            col_list_1.append((table_name, column_name))
            col_list_2.append((table_name_2, column_name))

    info_1, left_list_1, right_list_1 = getData(cn2id, col_list_1, left_list, right_list)
    info_2, left_list_2, right_list_2 = getData(cn2id, col_list_2, left_list, right_list)
    return TensorDataset(info_1, *left_list_1, *right_list_1, info_2, *left_list_2, *right_list_2)

def getTestData(dataset, left_list, right_list):
    tableDict, cn2id, id2cn = get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')
    col_list = []
    for table_name, columnList in tableDict.items():
        for column_name in columnList:
            col_list.append((table_name, column_name))

    info, left_list, right_list = getData(cn2id, col_list, left_list, right_list)
    return TensorDataset(info, *left_list, *right_list)