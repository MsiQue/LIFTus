import os
import time
import torch
import torch.nn as nn
import pickle
import random
from collections import defaultdict
from global_info import get_base_info
from loss import info_nce_loss
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup

def train(F, model, Data, optimizer, batch_size, n_epochs, device):
    dataloader = DataLoader(Data, batch_size, shuffle=True)
    num_steps = (len(Data) // batch_size) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)
    for epoch in range(1, n_epochs+1):
        model.train()
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = F(model, batch, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 10 == 0:
                print(f"step: {i}, loss: {loss.item()}")
            del loss

def inference_model(F, model, Data, batch_size, device):
    test_loader = DataLoader(Data, batch_size=batch_size, shuffle=False)
    model.eval()
    res = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = F(model, batch, device)
            if isinstance(outputs, tuple):
                res.append([o.detach().cpu() for o in outputs])
            else:
                res.append([outputs.detach().cpu()])

            del batch
            del outputs
            torch.cuda.empty_cache()


    L = len(res[0])
    return [torch.cat([t[i] for t in res], dim = 0) for i in range(L)]

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
    # digits_idx = [idx for idx, char in enumerate(s) if char.isdigit()]  #  '³'
    digits_idx = [idx for idx, char in enumerate(s) if (ord('0') <= ord(char) and ord(char) <= ord('9'))]
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

def adjust_length(tensor, d):
    current_length = tensor.size(0)
    if current_length < d:
        tensor = torch.cat([tensor, torch.zeros((d - current_length, tensor.size(1)), dtype=tensor.dtype)])
    elif current_length > d:
        tensor = tensor[:d]
    return tensor

def string_to_onehot_matrix(input_string, padding_len = 32):
    alphabet = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # 第一个是空格
    char_to_index = {char: i for i, char in enumerate(alphabet)}
    sequence_length = len(input_string)
    onehot_matrix = torch.zeros((sequence_length, len(alphabet) + 3))

    unknown_char_list = set()
    for i, char in enumerate(input_string):
        onehot_matrix[i, 0] = char.isdigit()
        onehot_matrix[i, 1] = char.isupper()
        onehot_matrix[i, 2] = char.islower()
        if char in char_to_index:
            onehot_matrix[i, 3 + char_to_index[char]] = 1
        else:
            unknown_char_list.add(char)

    return adjust_length(onehot_matrix, padding_len), unknown_char_list

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, criterion):
        super(SiameseNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.criterion = criterion

    def inference(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]

    def forward(self, input1, input2):
        output1 = self.inference(input1)
        output2 = self.inference(input2)
        z = torch.cat([output1, output2], dim=0)
        logits, labels = info_nce_loss(z, len(z) // 2, 2)
        loss = self.criterion(logits, labels)
        return loss

def getData(dataset, func_list, cn2id, device, n_sample = 10, n_check = 10):
    save_path_root = 'step_result/pattern'
    save_path = os.path.join(save_path_root, f'{dataset}_pattern_{n_sample}_{n_check}.pickle')
    pattern_dict = pickle.load(open(save_path, 'rb'))
    pattern_vec_list = [[] for _ in range(len(func_list))]
    cn_id_list = []
    pattern_id_list = []
    for table_name, table in pattern_dict.items():
        for column_name, column_patterns in table.items():
            if (table_name, column_name) not in cn2id:
                print(table_name, column_name)
                continue
            for i, pattern in enumerate(column_patterns):
                cn_id_list.append(cn2id[(table_name, column_name)])
                pattern_id_list.append(i)
                for i, func in enumerate(func_list):
                    pattern_vec_list[i].append(string_to_onehot_matrix(func(pattern))[0])
    return torch.tensor(cn_id_list).to(device), torch.tensor(pattern_id_list).to(device), [torch.stack(x, dim=0).to(device) for x in pattern_vec_list]

def F(model, batch, device):
    batch_ori, batch_arg = batch
    return model(batch_ori.to(device), batch_arg.to(device))

def train_pattern_encoder(dataset, trainData, hidden_size, lr, batchsize, device, n_epochs = 10):
    model_save_path_root = 'step_result/pattern_model'
    if not os.path.exists(model_save_path_root):
        os.makedirs(model_save_path_root)
    model_save_path = os.path.join(model_save_path_root, f'{dataset}_pattern_model_{hidden_size}_learningrate_{str(lr).replace(".", "_")}.pth')

    model = SiameseNetwork(input_size=98, hidden_size=hidden_size, criterion=nn.CrossEntropyLoss().to(device))
    model = model.to(device)

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        return model

    optimizer = AdamW(model.parameters(), lr=lr)
    train(F, model, trainData, optimizer, batchsize, n_epochs, device)

    torch.save(model.state_dict(), model_save_path)

    return model

def get_embs(id2cn, model, testData, device):
    raw_dict = defaultdict(list)
    output = inference_model(lambda model, batch, device : model.inference(batch[-1].to(device)), model, testData, batch_size=128, device=device)[0]

    for i, d in enumerate(testData):
        table_name, column_name = id2cn[d[0]]
        pattern_id = d[1].detach().cpu().numpy()
        emb = output[i].detach().cpu().numpy()
        raw_dict[(table_name, column_name)].append((table_name, column_name, pattern_id, emb))

    result_dict = defaultdict(dict)
    for (table_name, column_name), value in raw_dict.items():
        result_dict[table_name][column_name] = value

    return result_dict

def get_pattern_emb(dataset, hidden_size, lr, batchsize, device):
    save_path_root = 'embeddings/pattern'
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    save_path = os.path.join(save_path_root, f'{dataset}_pattern_emb.pickle')

    model_save_path_root = 'step_result/pattern_model'
    if not os.path.exists(model_save_path_root):
        os.makedirs(model_save_path_root)
    model_save_path = os.path.join(model_save_path_root, f'{dataset}_pattern_encoder.pth')

    if os.path.exists(save_path) and os.path.exists(model_save_path):
        print(f'Already complete get_pattern_emb on {save_path} before!')
        return

    start_time = time.time()
    print(f'Start get_pattern_emb on {dataset}.')

    tableDict, cn2id, id2cn = get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')
    cn_id_list, pattern_id_list, pattern_vec = getData(dataset, [lambda x: x, argument], cn2id, torch.device('cpu'))
    pattern_ori_vec = pattern_vec[0]
    pattern_arg_vec = pattern_vec[1]
    trainData = TensorDataset(pattern_ori_vec, pattern_arg_vec)
    testData = TensorDataset(cn_id_list, pattern_id_list, pattern_ori_vec)
    model = train_pattern_encoder(dataset, trainData, hidden_size, lr, batchsize, device)

    torch.save(model.state_dict(), model_save_path)

    res = get_embs(id2cn, model, testData, device)
    pickle.dump(res, open(save_path, 'wb'))

    print(f'Complete get_pattern_emb on {dataset} using {time.time() - start_time} s.')

def get_pattern_emb_eval(train_dataset, test_dataset, hidden_size, device):
    save_path_root = 'embeddings/pattern'
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    save_path = os.path.join(save_path_root, f'{test_dataset}_pattern_emb.pickle')

    if os.path.exists(save_path):
        print(f'Already complete get_pattern_emb_eval on {save_path} before!')
        return

    start_time = time.time()
    print(f'Start get_pattern_emb_eval on (train_dataset {train_dataset}) (test_dataset {test_dataset}) .')

    tableDict, cn2id, id2cn = get_base_info(test_dataset, f'base_info/{test_dataset}_base_info.pickle')

    cn_id_list, pattern_id_list, pattern_vec = getData(test_dataset, [lambda x: x], cn2id, torch.device('cpu'))
    pattern_ori_vec = pattern_vec[0]
    testData = TensorDataset(cn_id_list, pattern_id_list, pattern_ori_vec)

    model = SiameseNetwork(input_size=98, hidden_size=hidden_size, criterion=nn.CrossEntropyLoss().to(device))
    model = model.to(device)
    model_save_path = os.path.join('step_result/pattern_model', f'{train_dataset}_pattern_encoder.pth')
    model.load_state_dict(torch.load(model_save_path))

    res = get_embs(id2cn, model, testData, device)
    pickle.dump(res, open(save_path, 'wb'))

    print(f'Complete get_pattern_emb_eval on (train_dataset {train_dataset}) (test_dataset {test_dataset}) using {time.time() - start_time} s.')

if __name__ == '__main__':
    hidden_size = 256
    lr = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    get_pattern_emb('test', hidden_size, lr, 128, device)
    get_pattern_emb('test_split_2', hidden_size, lr, 128, device)

    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            for n3 in ['', '_split_2']:
                dataset = n1 + n2 + n3
                get_pattern_emb(dataset, hidden_size, lr, 128, device)