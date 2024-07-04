from collections import defaultdict
import os
import io
import torch
import pickle
import numpy as np
import pandas as pd

def delEmpty(values_raw):
    non_empty_series = values_raw.dropna()
    return non_empty_series

def getWordsCount(counts):
    words_counts = defaultdict(int)
    for k, v in counts.items():
        # for x in str(k).split():
        for x in split_text_by_non_alpha(str(k)):
            words_counts[x] += v
    words_counts = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    return words_counts

def is_int(x): # is_int('310246_1501') return True
    x = str(x)
    try:
        _ = int(x)
        return True
    except ValueError:
        return False

def is_float(x):
    x = str(x)
    try:
        _ = float(x)
        return True
    except ValueError:
        return False

def getColNum(x):
    x = x.split('____')[1]
    x = x[1:].split('_')[0]
    return int(x)

# def getColNum_1(x, table_path):
#     import pandas as pd
#     table_path = os.path.join(table_path, x)
#     df = pd.read_csv(table_path)
#     return df.shape[1]

def get_labels(tableDict, tn2id):
    labels = [-1 for i in range(len(tn2id))]
    M = {}
    for x, y in tableDict.items():
        tid = tn2id[x]
        tmp = x.split('____')[0]
        if tmp not in M:
            M[tmp] = len(M)
        labels[tid] = M[tmp]
    return labels

def get_labels_1(tableDict, tn2id):
    labels = [-1 for i in range(len(tn2id))]
    M = {}
    for x, y in tableDict.items():
        tid = tn2id[x]
        tmp = x[:5]
        if tmp not in M:
            M[tmp] = len(M)
        labels[tid] = M[tmp]
    return labels

def getLabels(tn2id):
    labels = [-1 for i in range(len(tn2id))]
    M = {}
    for x in tn2id.keys():
        tid = tn2id[x]
        tmp = x.split('____')[0]
        if tmp not in M:
            M[tmp] = len(M)
        labels[tid] = M[tmp]
    return labels

def getLabels_1(tn2id):
    labels = [-1 for i in range(len(tn2id))]
    M = {}
    for x in tn2id.keys():
        tid = tn2id[x]
        tmp = x[:5]
        if tmp not in M:
            M[tmp] = len(M)
        labels[tid] = M[tmp]
    return labels

def calc(s1, s2):
    a = s1 & s2
    b = s1 | s2
    if len(b) == 0:
        return 0
    return len(a) / len(b)

def get0(a):
    res = set()
    for x in a:
        res.add(x[0])
    return res

def calc_similarity_2col_words(col1, col2):
    return calc(get0(col1.words_topK), get0(col2.words_topK))

def calc_similarity_2col_cells(col1, col2):
    return calc(get0(col1.cells_topK), get0(col2.cells_topK))

def calc_similarity_2col(col1, col2):
    if (col1.pattern is not None) and (col2.pattern is not None):
        if col1.pattern == col2.pattern:
            return 0.8
    if len(prefix(col1.prefix, col2.prefix)) >= 2:
        return 0.8
    if len(suffix(col1.suffix, col2.suffix)) >= 2:
        return 0.8
    return max(calc_similarity_2col_cells(col1, col2), calc_similarity_2col_words(col1, col2))

def bidDiffer(a, b):
    s = 0
    for x, y in zip(a, b):
        s += abs(ord(x) - ord(y))
    return s

def getBucketID(col):
    return col.feat['classification'].tag

def merge(s1, s2):
    res = ''
    for x, y in zip(s1, s2):
        if x == '*' or y == '*':
            res += '*'
        elif x != y:
            res += '*'
        else:
            res += x
    return res

def getPattern(v):
    values = v.reset_index(drop=True)
    Len = len(values)
    if Len == 0:
        return None
    # if '*' in str(values[0]):
    #     print(col.belong_to, col.name, 0)
    pattern = str(values[0])
    for i in range(1, Len):
        # if '*' in str(values[i]):
        #     print(col.belong_to, col.name, i)
        pattern = merge(pattern, str(values[i]))
        if pattern == '*' * len(str(values[i])):
            return None
    return pattern

def prefix(s1, s2):
    res = ''
    for x, y in zip(s1, s2):
        if x == y:
            res += x
        else:
            break
    return res

def getPrefix(v):
    values = v.reset_index(drop=True)
    Len = len(values)
    if Len == 0:
        return ''
    res = str(values[0])
    for i in range(1, Len):
        res = prefix(res, str(values[i]))
        if res == '':
            return ''
    return res

def suffix(s1, s2):
    return prefix(s1[::-1], s2[::-1])[::-1]

def getSuffix(v):
    values = v.reset_index(drop=True)
    Len = len(values)
    if Len == 0:
        return ''
    res = str(values[0])
    for i in range(1, Len):
        res = suffix(res, str(values[i]))
        if res == '':
            return ''
    return res

import re

def split_text_by_non_alpha(text):
    words = re.split(r'[^a-zA-Z]+', text)
    words = [word for word in words if word]
    return words

def checkPattern(p1, p2):
    t = (p1 is None) + (p2 is None)
    if t == 2:
        return True
    elif t == 1:
        return False
    if len(p1) != len(p2):
        return False
    for x, y in zip(p1, p2):
        if x == '*' or y == '*':
            continue
        if x != y:
            return False
    return True

def checkPrefix(a, b):
    if len(a) * len(b) == 0:
        if len(a) + len(b) != 0:
            return False
    if a.startswith(b):
        return True
    if b.startswith(a):
        return True
    return False

def checkSuffix(a, b):
    if len(a) * len(b) == 0:
        if len(a) + len(b) != 0:
            return False
    if a.endswith(b):
        return True
    if b.endswith(a):
        return True
    return False

def getVector(col):
    return col.feat['classification'].feat_value + col.feat['distance'].feat_value

def compareColumn(c1, c2):
    res = []
    res.append(1 if c1.feat['classification'].tag == c2.feat['classification'].tag else 0)
    res.append(1 if checkPattern(c1.pattern, c2.pattern) else 0)
    res.append(1 if checkPrefix(c1.prefix, c2.prefix) else 0)
    res.append(1 if checkSuffix(c1.suffix, c2.suffix) else 0)
    return res

def getColumnPairVec(col1, col2):
    v = [(x - y) ** 2 for x, y in zip(getVector(col1), getVector(col2))]
    return v + compareColumn(col1, col2) + [calc_similarity_2col_words(col1, col2), calc_similarity_2col_cells(col1, col2)]

def getColumnPairInfo(col1, col2):
    feat_1 = getVector(col1)
    feat_2 = getVector(col2)
    share_info = compareColumn(col1, col2) + [calc_similarity_2col_words(col1, col2), calc_similarity_2col_cells(col1, col2)]
    return feat_1, feat_2, share_info

def getEmb(fasttext_info, w):
    if w in fasttext_info:
        return fasttext_info[w]
    else:
        return np.zeros(300)

def getVector_word(word_List, fasttext_info):
    if len(word_List) == 0:
        return np.zeros(300)
    res = []
    for w in word_List:
        res.append(getEmb(fasttext_info, w))
    return np.sum(res, axis=0)

def _(a, b):
    if b in a:
        return a[b]
    if b.endswith('\r'):
        return a[b[:-1]]
    return a[b+'\r']

def getColPairVectors(tfidf_topk_save_path, topk, fasttext_info, col1_list, col2_list, label):
    from dataset import ColumnPairs
    feat_1_list = []
    feat_2_list = []
    word_1_list = []
    word_2_list = []
    share_info_list = []
    y = []
    info_list = []
    tfidf_info = pickle.load(open(tfidf_topk_save_path, 'rb'))
    for c1, c2 in zip(col1_list, col2_list):
        feat_1, feat_2, share_info = getColumnPairInfo(c1, c2)
        word_1 = getVector_word(_(tfidf_info[c1.belong_to], c1.name)[:topk], fasttext_info)
        word_2 = getVector_word(_(tfidf_info[c2.belong_to], c2.name)[:topk], fasttext_info)
        info = [(c1.belong_to, c1.name), (c2.belong_to, c2.name)]
        feat_1_list.append(feat_1)
        feat_2_list.append(feat_2)
        word_1_list.append(word_1)
        word_2_list.append(word_2)
        share_info_list.append(share_info)
        y.append(label)
        info_list.append(info)
    return ColumnPairs(feat_1_list, feat_2_list, word_1_list, word_2_list, share_info_list, y, info_list)

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data

def get_topk(model, feature_names, sentence, k):
    tfidf_values = model.transform([sentence])
    topk_indices = tfidf_values.indices[np.argsort(tfidf_values.data)[-k:][::-1]]
    topk_words = [feature_names[index] for index in topk_indices]
    return topk_words

def get_col_topk(model, feature_names, col, k):
    sentence = col.astype(str).str.cat(sep=' ')
    return get_topk(model, feature_names, sentence, k)

def solve_table(vectorizer, feature_names, table_path, k):
    df = pd.read_csv(table_path, lineterminator='\n')
    res = {}
    for col_name in list(df.columns):
        res[col_name] = get_col_topk(vectorizer, feature_names, df[col_name], k)
    table_name = table_path.split('/')[-1]
    # print(f'Finish {table_name}')
    return table_name, res

def string_to_onehot_matrix(input_string):
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

    return onehot_matrix, unknown_char_list

def revert(M):
    def f(a):
        a = a[3:]
        if torch.all(a == 0):
            return "[UNSEEN CHAR]"
        idx = (a == 1).nonzero(as_tuple=False).item()
        return " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[idx]
    s = ""
    for i in range(M.shape[0]):
        s += f(M[i])
    return s

if __name__ == '__main__':
    pass