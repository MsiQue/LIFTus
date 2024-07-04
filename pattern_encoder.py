import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from torch.utils.data import DataLoader, Dataset

def padding(batch):
    col1_pattern_list = [sample['col1_pattern_list'] for sample in batch]
    col2_pattern_list = [sample['col2_pattern_list'] for sample in batch]
    col1_pattern_list = torch.nn.utils.rnn.pad_sequence(col1_pattern_list, batch_first=True)
    col2_pattern_list = torch.nn.utils.rnn.pad_sequence(col2_pattern_list, batch_first=True)
    return {'col1_pattern_list': col1_pattern_list, 'col2_pattern_list': col2_pattern_list, 'y': torch.tensor([sample['y'] for sample in batch]), 'info': [sample['info'] for sample in batch]}

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def inference(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, input1, input2):
        output1 = self.inference(input1)
        output2 = self.inference(input2)
        return output1, output2

def train_siamese_network(siamese_net, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        batch_cnt = 0
        for batch in dataloader:
            input1 = batch['col1_pattern_list']
            input2 = batch['col2_pattern_list']
            label = batch['y']
            optimizer.zero_grad()
            output1, output2 = siamese_net(input1, input2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            batch_cnt += 1
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_cnt, loss))

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.PairwiseDistance()(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def trainprocess(pos_sample_path, neg_sample_path, nn_model_save_path, hidden_size, learning_rate, num_epochs):
    pos_sample = pickle.load(open(pos_sample_path, 'rb'))
    neg_sample = pickle.load(open(neg_sample_path, 'rb'))
    train_data = pos_sample + neg_sample

    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=padding)

    net = SiameseNetwork(input_size=98, hidden_size=hidden_size)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_siamese_network(net, train_dataloader, criterion, optimizer, num_epochs)
    torch.save(net.state_dict(), nn_model_save_path)

def strReplace(s, idxList, newChar):
    for idx in idxList:
        s = s[:idx] + newChar + s[idx + 1:]
    return s

def argument(s):
    result = []
    non_star_chars_idx = [idx for idx, char in enumerate(s) if char != '*']

    # 随机选择一个非*的字符，将其替换为*
    if non_star_chars_idx:
        random_idx = random.choice(non_star_chars_idx)
        result.append(strReplace(s, [random_idx], '*'))

    # 随机选择两个非*的字符，将其替换为*
    if len(non_star_chars_idx) >= 2:
        random_idx = random.sample(non_star_chars_idx, 2)
        result.append(strReplace(s, random_idx, '*'))

    # 将其中一个数字替换为另一个数字
    digits_idx = [idx for idx, char in enumerate(s) if char.isdigit()]
    if digits_idx:
        random_idx = random.choice(digits_idx)
        _new = (int(s[random_idx]) + random.randint(1, 9)) % 10
        result.append(s[:random_idx] + str(_new) + s[random_idx + 1:])

    # 将其中一个小写字母替换为另一个小写字母
    lowercase_letters_idx = [idx for idx, char in enumerate(s) if char.islower()]
    if lowercase_letters_idx:
        random_idx = random.choice(lowercase_letters_idx)
        _new = (ord(s[random_idx]) - 97 + random.randint(1, 25)) % 26 + 97
        result.append(s[:random_idx] + chr(_new) + s[random_idx + 1:])

    # 将其中一个大写字母替换为另一个大写字母
    uppercase_letters_idx = [idx for idx, char in enumerate(s) if char.isupper()]
    if uppercase_letters_idx:
        random_idx = random.choice(uppercase_letters_idx)
        _new = (ord(s[random_idx]) - 65 + random.randint(1, 25)) % 26 + 65
        result.append(s[:random_idx] + chr(_new) + s[random_idx + 1:])

    if result:
        return random.choice(result)
    else:
        return s

if __name__ == '__main__':
    cnt = {'TUS_small' : 3102, 'TUS_large' : 9926, 'SANTOS_small' : 1635, 'SANTOS_large' : 46710}

    # times = 100
    # # for dataset in ['TUS_small', 'TUS_large', 'SANTOS_small', 'SANTOS_large']:
    # # for dataset in ['TUS_small', 'TUS_large', 'SANTOS_small']:
    # for dataset in ['SANTOS_small', 'TUS_large']:
    #     pos_sample_path = f'pos_neg_sample_pattern/pos_sample_argument_{dataset}_0.1_{cnt[dataset]}_patterns_{times}_times.pth'
    #     neg_sample_path = f'pos_neg_sample_pattern/neg_sample_argument_{dataset}_0.1_{cnt[dataset] * times}_times.pth'
    #     for hidden_size in [32, 64, 128, 256]:
    #         for lr in [0.001, 0.0005]:
    #             nn_model_save_path = f'nn/pattern_emb/{dataset}/nn_model_{dataset}_pattern_lstm_contrastive_loss_times_{times}_hiddensize_{hidden_size}_learningrate_{str(lr).replace(".", "_")}.pth'
    #             trainprocess(pos_sample_path, neg_sample_path, nn_model_save_path, hidden_size, lr, num_epochs=5)



    dataset = 'SANTOS_large'
    times = 10
    num_epochs = 3
    pos_sample_path = f'pos_neg_sample_pattern/pos_sample_argument_{dataset}_0.1_{cnt[dataset]}_patterns_{times}_times.pth'
    neg_sample_path = f'pos_neg_sample_pattern/neg_sample_argument_{dataset}_0.1_{cnt[dataset] * times}_times.pth'
    # for hidden_size in [32, 64, 128, 256]:
    for hidden_size in [256]:
        for lr in [0.0005]:
            if hidden_size == 32 and lr > 0.006:
                continue
            nn_model_save_path = f'nn/pattern_emb/{dataset}/nn_model_{dataset}_pattern_lstm_contrastive_loss_times_{times}_hiddensize_{hidden_size}_learningrate_{str(lr).replace(".", "_")}.pth'
            trainprocess(pos_sample_path, neg_sample_path, nn_model_save_path, hidden_size, lr, num_epochs)

    # cnn = CNN()
    # siamese_net = SiameseNetwork(cnn)
    # criterion = ContrastiveLoss()
    # optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
    #
    # train_siamese_network(siamese_net, train_dataloader, criterion, optimizer, num_epochs=5)
