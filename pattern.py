from table import *
from dataset import ColPattern
from utils import _
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from train_test import train

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def inference(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, x1, x2):
        x1 = self.inference(x1)
        x2 = self.inference(x2)
        distance = (x1 - x2) ** 2
        output = self.fc(distance)
        output = self.sigmoid(output)
        return output

def getPattern(dataset):
    save_path = 'table_info/' + dataset + '_table_info.pickle'
    table_info = pickle.load(open(save_path, 'rb'))

    res = []
    for table_name, table in tqdm.tqdm(table_info.items()):
        for col_name, col in table.columnDict.items():
            if col.pattern is None:
                continue
            res.append((col.pattern, table_name, col_name))
    return res

def genPositive_pattern_all(dataset, table_info, table_info_split_2, sample_cnt):
    pos_sample_save_path = 'pos_neg_sample_pattern/pos_sample_all_pattern_' + dataset + '.pth'
    if os.path.exists(pos_sample_save_path):
        return pickle.load(open(pos_sample_save_path, 'rb'))

    col1_pattern_list = []
    col2_pattern_list = []
    info = []
    for table_name, table in tqdm.tqdm(table_info.items()):
        if table.row_len < 2:
            continue
        for column_name in table.columnDict.keys():
            for i in range(1, sample_cnt + 1):
                for j in range(i + 1, sample_cnt + 1):
                    try:
                        col1 = _(table_info_split_2[table_name.split('.')[0] + f'____{i}.csv'].columnDict, column_name)
                        col2 = _(table_info_split_2[table_name.split('.')[0] + f'____{j}.csv'].columnDict, column_name)
                        col1_pattern = col1.pattern
                        col2_pattern = col2.pattern
                        if col1_pattern is None:
                            continue
                        if col2_pattern is None:
                            continue
                        col1_pattern_list.append(col1_pattern)
                        col2_pattern_list.append(col2_pattern)
                        info.append([(col1.belong_to, col1.name), (col2.belong_to, col2.name)])
                    except:
                        print(table_name, i, j, '='*60)

    res = ColPattern(col1_pattern_list, col2_pattern_list, 1, info)
    pickle.dump(res, open(pos_sample_save_path, 'wb'))
    return res

def genNegative_step(candidate, table_info):
    while True:
        _, x, a = random.choice(candidate)
        _, y, b = random.choice(candidate)
        # 纯随机选，不强制表名前缀不同，只要求表不同
        if x == y:
            continue
        return table_info[x].columnDict[a], table_info[y].columnDict[b]

def genNegative(dataset, cnt, table_info):
    neg_sample_save_path = 'pos_neg_sample_pattern/neg_sample_pattern' + dataset + '_' + str(cnt) + '.pth'
    if os.path.exists(neg_sample_save_path):
        return pickle.load(open(neg_sample_save_path, 'rb'))

    candidate = getPattern(dataset)
    col1_pattern_list = []
    col2_pattern_list = []
    info = []
    for i in tqdm.tqdm(range(cnt)):
        col1, col2 = genNegative_step(candidate, table_info)
        col1_pattern_list.append(col1.pattern)
        col2_pattern_list.append(col2.pattern)
        info.append([(col1.belong_to, col1.name), (col2.belong_to, col2.name)])
    res = ColPattern(col1_pattern_list, col2_pattern_list, 0, info)
    pickle.dump(res, open(neg_sample_save_path, 'wb'))
    return res

def F(model, batch):
    col1_pattern_list = batch['col1_pattern_list']
    col2_pattern_list = batch['col2_pattern_list']
    return model(col1_pattern_list, col2_pattern_list)

def padding(batch):
    col1_pattern_list = [sample['col1_pattern_list'] for sample in batch]
    col2_pattern_list = [sample['col2_pattern_list'] for sample in batch]
    col1_pattern_list = torch.nn.utils.rnn.pad_sequence(col1_pattern_list, batch_first=True)
    col2_pattern_list = torch.nn.utils.rnn.pad_sequence(col2_pattern_list, batch_first=True)
    return {'col1_pattern_list': col1_pattern_list, 'col2_pattern_list': col2_pattern_list, 'y': torch.tensor([sample['y'] for sample in batch]), 'info': [sample['info'] for sample in batch]}

def trainLstm(dataset, pos_sample_path, neg_sample_path, eval_data_path):
    if dataset == 'SANTOS_small':
        epochs = 3
    elif dataset == 'TUS_small':
        epochs = 6
    elif dataset == 'TUS_large':
        epochs = 6
    elif dataset == 'SANTOS_large':
        epochs = 6

    pos_sample = pickle.load(open(pos_sample_path, 'rb'))
    neg_sample = pickle.load(open(neg_sample_path, 'rb'))
    train_data = pos_sample + neg_sample
    eval_data = pickle.load(open(eval_data_path, 'rb'))

    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=padding)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=True, collate_fn=padding)

    net = LSTMModel(input_size = 98, hidden_size = 32)
    # criterion = nn.BCELoss(weight = torch.tensor([1.0, 15.0]))
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    nn_model_save_path = f'nn/{dataset}/nn_model_{dataset}_lstm.pth'
    train(F, net, train_dataloader, eval_dataloader, criterion, optimizer, nn_model_save_path, epochs)

    # nn_model_save_path = 'nn/nn_model_' + dataset + '.pth'
    # torch.save(net.state_dict(), nn_model_save_path)

if __name__ == '__main__':
    # res = sorted(getPattern('TUS_small'))
    # unknown_char_set = set()
    # for i, r in enumerate(res):
    #     print("%-10d %-50s %-50s %-50s" % (i, r[0], r[1], r[2]))
    #     _, unknown_char = string_to_onehot_matrix(r[0])
    #     unknown_char_set |= unknown_char
    #
    # print(len(unknown_char_set))
    # print(unknown_char_set)

    # dataset = 'TUS_small'
    # dataset = 'SANTOS_small'
    # dataset = 'SANTOS_large'
    dataset = 'TUS_large'
    save_path = 'table_info/' + dataset + '_table_info.pickle'
    table_info = pickle.load(open(save_path, 'rb'))
    table_info_split_2 = pickle.load(open(f'table_info_split_2_times_3/{dataset}_table_info_split_2_times_3.pickle', 'rb'))
    genPositive_pattern_all(dataset, table_info, table_info_split_2, 6)

    if dataset == 'SANTOS_small':
        neg_sample_cnt = 16000
    elif dataset == 'TUS_small':
        neg_sample_cnt = 250000
    elif dataset == 'TUS_large':
        neg_sample_cnt = 850000
    elif dataset == 'SANTOS_large':
        neg_sample_cnt = 1800000

    genNegative(dataset, neg_sample_cnt, table_info)

    # pos_sample_path = 'pos_neg_sample_pattern/pos_sample_all_pattern_SANTOS_small.pth'
    # neg_sample_path = 'pos_neg_sample_pattern/neg_sample_patternSANTOS_small_16000.pth'
    # eval_data_path = 'eval_sample_pattern/sample_SANTOS_small_10000pos_10000neg_pattern.pth'
    pos_sample_path = f'pos_neg_sample_pattern/pos_sample_all_pattern_{dataset}.pth'
    neg_sample_path = f'pos_neg_sample_pattern/neg_sample_pattern{dataset}_{neg_sample_cnt}.pth'
    eval_data_path = f'eval_sample_pattern/sample_{dataset}_10000pos_10000neg_pattern.pth'

    trainLstm(dataset, pos_sample_path, neg_sample_path, eval_data_path)