import os
import time
import torch
import torch.nn as nn
from trainprocess import train
from transformers import AdamW
from model import Net
from preprocess import getTrainData, getTestData
from result import calc

def fill_template_list(L, s):
    res = []
    for x in L:
        if isinstance(x, str):
            res.append(x.format(s))
        elif isinstance(x, list):
            res.append([y.format(s) if isinstance(y, str) else y for y in x])
    return res
    # return [x.format(s) if isinstance(x, str) else x for x in L]

def F(model, batch):
    sep_pos = len(batch) // 2
    _batch1 = batch[:sep_pos]
    _batch2 = batch[sep_pos:]
    info_1, *batch1 = _batch1
    info_2, *batch2 = _batch2
    sep_pos = len(model.input_size_left)
    return model(batch1[:sep_pos], batch1[sep_pos:], batch2[:sep_pos], batch2[sep_pos:])

def run_step(test_dataset, trainData, testData, input_size, device, num_heads, hiddensize, lr, batchsize, model_save_path, n_epochs = 10):
    if model_save_path is not None:
        if os.path.exists(model_save_path):
            return
        print(f'Create {model_save_path}')
    model = Net(input_size_left=input_size[0], input_size_right=input_size[1], num_heads=num_heads, hidden_size=hiddensize, criterion=nn.CrossEntropyLoss().to(device), device=device)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train(F, model, trainData, optimizer, batchsize, n_epochs)
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
    dict_k = {'TUS_small': 60, 'TUS_large': 60, 'SANTOS_small': 10, 'SANTOS_large': 20, 'test': 5, 'OpenDataLarge' : 20}
    return calc(test_dataset, model, testData, dict_k[test_dataset], 'HNSW64')

def run(train_dataset, test_dataset, left_list, right_list, input_size, num_heads_list, hiddensize_list, lr_list, batchsize):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t1 = time.time()
    trainData = getTrainData(train_dataset + '_split_2', fill_template_list(left_list, train_dataset + '_split_2'), fill_template_list(right_list, train_dataset + '_split_2'))
    testData = getTestData(test_dataset, fill_template_list(left_list, test_dataset), fill_template_list(right_list, test_dataset))
    print(f'Load train dataset {train_dataset} and test dataset {test_dataset} cost {time.time() - t1} s')

    result = []
    for num_heads in num_heads_list:
        for hiddensize in hiddensize_list:
            for lr in lr_list:
                print(f'train at {train_dataset} test at {test_dataset}', num_heads, hiddensize, lr)
                res = run_step(test_dataset, trainData, testData, input_size, device, num_heads, hiddensize, lr, batchsize, None)
                print(res)
                result.append((f'train at {train_dataset} test at {test_dataset}', num_heads, hiddensize, lr, res))
    return result

def emb_path(L):
    emb_path_dict = {
        'statistic' : 'embeddings/statistic/{}_statistic.pickle',
        'paragraph' : ['embeddings/paragraph/{}_sample_32_paragraph__bert.pickle', 32, 768],
        'word' : ['embeddings/word/{}_word_emb_64_sample.pickle', 64, 300],
        'number': ['embeddings/number/{}_number_emb_128_dim.pickle', 14, 128],
        'pattern': ['embeddings/pattern/{}_pattern_emb.pickle', 128, 128]
    }
    return [emb_path_dict[x] for x in L]

def dims(L, num_heads = 8):
    dim_dict = {
        'statistic' : 99,
        'paragraph' : [768, num_heads],
        'word' : [300, num_heads],
        'number' : [128, num_heads],
        'pattern' : [128, num_heads]
    }
    return [dim_dict[x] for x in L]

if __name__ == '__main__':
    leftList = ['statistic']
    rightList = ['paragraph', 'word', 'number', 'pattern']
    # rightList = ['paragraph', 'word', 'number']
    # rightList = ['word', 'number']

    # run('test', emb_path(leftList), emb_path(rightList), [dims(leftList), dims(rightList)], [2], [128],[0.0005], 64)
    run('SANTOS_small', emb_path(leftList), emb_path(rightList), [dims(leftList), dims(rightList)], [2], [128], [0.0005], 64)

    result = []
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            dataset = n1 + n2
            # dataset_split_2 = n1 + n2 + '_split_2'
            result += run(dataset, emb_path(leftList), emb_path(rightList), [dims(leftList), dims(rightList)], [2], [128], [0.0005], 64)
    for r in result:
        print(r)