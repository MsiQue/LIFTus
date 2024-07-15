from global_info import get_csv_folder
import pandas as pd
import numpy as np
import pickle
import tqdm
import os
import io
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data

def word2vec_fasttext(fasttext_info, w):
    if w in fasttext_info:
        return fasttext_info[w]
    else:
        return np.zeros(300)

def word_encoder_by_table(args):
    table_path, number_topK, fasttext_info, string_k = args
    data = pd.read_csv(table_path, lineterminator='\n')
    res = {}
    for column_name in data.columns:
        # print(table_path, column_name, number_topK[column_name])
        L = sorted(list(number_topK[column_name]), key = lambda x: x[1], reverse=True)
        res[column_name] = [(x[0], x[1], word2vec_fasttext(fasttext_info, x[0])) for x in L]
    return res
def word_encoder_all(dataset, fasttext_info, string_k = 64, number_k = 512):
    csv_folder = get_csv_folder(dataset)
    tokenized_path = f'step_result/tokens/{dataset}_tokens_{string_k}_string_{number_k}_number.pickle'
    tokenized_info = pickle.load(open(tokenized_path, 'rb'))
    output_dict_file_root = 'embeddings/word'
    if not os.path.exists(output_dict_file_root):
        os.makedirs(output_dict_file_root)
    output_dict_file = os.path.join(output_dict_file_root, f'{dataset}_word_emb_{string_k}_sample.pickle')

    res = {}
    for table_name in tqdm.tqdm(os.listdir(csv_folder)):
        table_path = os.path.join(csv_folder, table_name)
        args = (table_path, tokenized_info[table_name][0], fasttext_info, string_k)
        res[table_name] = word_encoder_by_table(args)
    pickle.dump(res, open(output_dict_file, 'wb'))

if __name__ == '__main__':
    fasttext_path = 'LM/fasttext/wiki-news-300d-1M.vec'
    fasttext_info = load_vectors(fasttext_path)

    print('finishing loading fasttext')

    word_encoder_all('test', fasttext_info)
    word_encoder_all('test_split_2', fasttext_info)

    # for n1 in ['TUS_', 'SANTOS_']:
    #     for n2 in ['small', 'large']:
    #         for n3 in ['', '_split_2']:
    #             dataset = n1 + n2 + n3
    #             word_encoder_all(dataset, fasttext_info)