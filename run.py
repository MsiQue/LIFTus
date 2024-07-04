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

def run_step(dataset, trainData, testData, input_size, device, num_heads, hiddensize, lr, batchsize, model_save_path, n_epochs = 10):
    if model_save_path is not None:
        if os.path.exists(model_save_path):
            return
        print(f'Create {model_save_path}')
    model = Net(input_size_left=input_size[0], input_size_right=input_size[1], num_heads=num_heads, hidden_size=hiddensize, criterion=nn.CrossEntropyLoss().to(device))
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train(F, model, trainData, optimizer, batchsize, n_epochs)
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
    dict_k = {'TUS_small': 60, 'TUS_large': 60, 'SANTOS_small': 10, 'SANTOS_large': 20}
    return calc(dataset, model, testData, dict_k[dataset], 'HNSW64')

def run(dataset, left_list, right_list, input_size, message, num_heads_list, hiddensize_list, lr_list, batchsize):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t1 = time.time()
    trainData = getTrainData(dataset + '_split_2', fill_template_list(left_list, dataset + '_split_2'), fill_template_list(right_list, dataset + '_split_2'), device)
    testData = getTestData(dataset, fill_template_list(left_list, dataset), fill_template_list(right_list, dataset), device)
    print(f'Load dataset {dataset} cost {time.time() - t1} s')

    result = []
    for num_heads in num_heads_list:
        for hiddensize in hiddensize_list:
            for lr in lr_list:
                # model_save_path = f'nn/{dataset}/{message}/{dataset}_{message}_{num_heads}_heads_{hiddensize}_hiddensize_{str(lr).replace(".", "_")}_lr.pth'

                print(dataset, num_heads, hiddensize, lr)
                res = run_step(dataset, trainData, testData, input_size, device, num_heads, hiddensize, lr, batchsize, None)
                print(res)
                result.append((dataset, num_heads, hiddensize, lr, res))
    return result

if __name__ == '__main__':
    result = []
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            dataset = n1 + n2
            dataset_split_2 = n1 + n2 + '_split_2'

            # message = "word_feature"
            # message = "word"
            # message = "feature=aveword_feature_pattern"
            message = "feature=fasttextword_feature_pattern"

            emb_path_dict = {
                "feature=aveword_feature_pattern": (
                    ['/data/qiuermu/test_74/vectors/featureEmb/{}_featureEmb.pickle'],
                    ['/data/qiuermu/test_74/vectors/wordEmb/{}_wordEmb_top_10.pickle', '/data/qiuermu/test_74/vectors/featureEmb/{}_featureEmb.pickle', '/data/qiuermu/test_74/vectors/patternEmb/{}_patternEmb.pickle'],
                    [[11], [300, 11, 64]]
                ),
                "feature=fasttextword_feature_pattern": (
                    ['/data/qiuermu/test_74/vectors/featureEmb/{}_featureEmb.pickle'],
                    [['vectors/wordEmb_all/fasttext_{}_wordnijia_split.pickle', 200, 300], '/data/qiuermu/test_74/vectors/featureEmb/{}_featureEmb.pickle', '/data/qiuermu/test_74/vectors/patternEmb/{}_patternEmb.pickle'],
                    [[11], [300, 11, 64]]
                )
            }
            emb_path_tuple = emb_path_dict[message]
            # result += run(dataset, emb_path_tuple[0], emb_path_tuple[1], emb_path_tuple[2], message, [1, 2, 4, 8], [32, 64, 128, 256], [0.001, 0.0005])
            result += run(dataset, emb_path_tuple[0], emb_path_tuple[1], emb_path_tuple[2], message, [2], [128], [0.0005], 32)
    for r in result:
        print(r)