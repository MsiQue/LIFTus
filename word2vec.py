from collections import defaultdict
import numpy as np
import pickle
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

def word2vec(source_file, target_file, method = 'fasttext', **kwargs):
    if method == 'fasttext':
        model = kwargs['model']
    words_dict = pickle.load(open(source_file, 'rb'))
    result = defaultdict(dict)
    for table_name, table in words_dict.items():
        for column_name, word_counter in table.items():
            tmp = []
            for word, cnt in word_counter.items():
                if method == 'fasttext':
                    emb = word2vec_fasttext(model, word)
                else:
                    emb = None
                tmp.append((cnt, word, emb))
            result[table_name][column_name] = sorted(tmp, key=lambda x: x[0], reverse=True)
    pickle.dump(result, open(target_file, 'wb'))

if __name__ == '__main__':
    method = 'fasttext'
    fasttext_info = load_vectors('/data/qiuermu/fasttext/wiki-news-300d-1M.vec')
    # for file in os.listdir('tf'):
    #     if file.endswith('.pickle'):
    #         source_file = os.path.join('tf', file)
    #         target_file = 'vectors/wordEmb_all/' + method + '_' + file
    #         print(f'Processing {source_file}')
    #         word2vec(source_file, target_file, method='fasttext', model = fasttext_info)

    word2vec('tf/SANTOS_large_split_2_wordnijia_split.pickle', 'vectors/wordEmb_all/SANTOS_large_split_2_wordnijia_split.pickle', method='fasttext', model=fasttext_info)
    word2vec('tf/SANTOS_large_wordnijia_split.pickle', 'vectors/wordEmb_all/SANTOS_large_wordnijia_split.pickle', method='fasttext', model=fasttext_info)
