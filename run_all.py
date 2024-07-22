from global_info import get_base_info
from table_split import random_split
from statistic_driven_aspect_weighter import column_statistic_all
from tokenizer import tokenizer_all, tokenizer_all_parallel
from number_encoder import number_encoder_all
from pattern_summarizer import summarize_Pattern
from paragraph_encoder import paragraph_encoder_all, convertFunc_bert
from transformers import BertTokenizer, BertModel
from word_encoder import word_encoder_all, load_vectors
from pattern_encoder import get_pattern_emb
from run import run, emb_path, dims
import nltk
import torch

def run_all(dataset, leftList, rightList, device, sample_cnt = 200, string_k = 64, number_k = 512, number_dim = 128, n_sample = 10, n_check = 10, hidden_size = 256, lr = 0.005, batchsize = 128):
    random_split(dataset, 1)
    split_suffix = '_split_2'
    get_base_info(dataset, f'base_info/{dataset}_base_info.pickle')

    column_statistic_all(dataset + split_suffix, sample_cnt)
    tokenizer_all(dataset + split_suffix, string_k, number_k)
    number_encoder_all(dataset + split_suffix, number_dim, string_k, number_k)

    fasttext_path = 'LM/fasttext/wiki-news-300d-1M.vec'
    fasttext_info = load_vectors(fasttext_path)
    word_encoder_all(dataset + split_suffix, fasttext_info)

    LM_path = 'LM/bert'
    tokenizer = BertTokenizer.from_pretrained(LM_path)
    model = BertModel.from_pretrained(LM_path).to(device)
    paragraph_encoder_all(dataset + split_suffix, 128, 'bert', is_parallel=True, parallel_cnt=1, sequential_limit=30, convertFunc=convertFunc_bert, tokenizer=tokenizer, model=model, device=device)

    summarize_Pattern(dataset + split_suffix, n_sample, n_check)
    get_pattern_emb(dataset + split_suffix, hidden_size, lr, batchsize, device)

    return run(dataset, emb_path(leftList), emb_path(rightList), [dims(leftList), dims(rightList)], [2], [128], [0.0005], 64)

if __name__ == '__main__':
    leftList = ['statistic']
    rightList = ['paragraph', 'word', 'number', 'pattern']

    nltk.download('words')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_all('test', leftList, rightList, device)

    result = []
    for n1 in ['TUS_', 'SANTOS_']:
        for n2 in ['small', 'large']:
            dataset = n1 + n2
            result += run_all(dataset, leftList, rightList, device)
    for r in result:
        print(r)