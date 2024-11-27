from global_info import get_base_info
from table_split import random_split
from statistic_driven_aspect_weighter import column_statistic_parallel
from tokenizer import tokenizer_all_parallel
from number_encoder import number_encoder_all
from pattern_summarizer import summarize_Pattern
from paragraph_encoder import paragraph_encoder_all, convertFunc_bert, tokenizerFunc
from transformers import BertTokenizer, BertModel
from word_encoder import word_encoder_all, load_vectors
from pattern_encoder import get_pattern_emb, get_pattern_emb_eval
from run import run, emb_path, dims
import nltk
import torch
import csv

def run_all(train_dataset, test_dataset, leftList, rightList, device, sample_cnt = 200, string_k = 64, number_k = 512, number_dim = 128, n_sample = 10, n_check = 10, hidden_size_pattern_encoder = 128, lr_pattern_encoder = 0.005, batchsize_pattern_encoder = 128, sequential_cnt=10, parallel_cnt=30):
    csv.field_size_limit(100000000)
    random_split(train_dataset, 1)
    split_suffix = '_split_2'

    column_statistic_parallel(train_dataset + split_suffix, sample_cnt, sequential_cnt, parallel_cnt)
    column_statistic_parallel(test_dataset, sample_cnt, sequential_cnt, parallel_cnt)

    tokenizer_all_parallel(train_dataset + split_suffix, string_k, number_k, sequential_cnt, parallel_cnt)
    tokenizer_all_parallel(test_dataset, string_k, number_k, sequential_cnt, parallel_cnt)

    number_encoder_all(train_dataset + split_suffix, number_dim, string_k, number_k)
    number_encoder_all(test_dataset, number_dim, string_k, number_k)

    fasttext_path = 'LM/fasttext/wiki-news-300d-1M.vec'
    fasttext_info = load_vectors(fasttext_path)
    word_encoder_all(train_dataset + split_suffix, fasttext_info)
    word_encoder_all(test_dataset, fasttext_info)

    LM_path = 'LM/bert'
    tokenizer = BertTokenizer.from_pretrained(LM_path)
    model = BertModel.from_pretrained(LM_path).to(device)
    paragraph_encoder_all(train_dataset + split_suffix, 32, 'bert', is_parallel=True, parallel_cnt=30, sequential_limit=30, tokenizerFunc=tokenizerFunc, tokenizer=tokenizer, model=model, device=device)
    paragraph_encoder_all(test_dataset, 32, 'bert', is_parallel=True, parallel_cnt=30, sequential_limit=30, tokenizerFunc=tokenizerFunc, tokenizer=tokenizer, model=model, device=device)
    # paragraph_encoder_all(train_dataset + split_suffix, 128, 'bert', is_parallel=False, convertFunc=convertFunc_bert, tokenizer=tokenizer, model=model, device=device)
    # paragraph_encoder_all(test_dataset, 128, 'bert', is_parallel=False, convertFunc=convertFunc_bert, tokenizer=tokenizer, model=model, device=device)

    summarize_Pattern(train_dataset + split_suffix, n_sample, n_check, True, sequential_cnt, parallel_cnt)
    summarize_Pattern(test_dataset, n_sample, n_check, True, sequential_cnt, parallel_cnt)

    get_pattern_emb(train_dataset + split_suffix, hidden_size_pattern_encoder, lr_pattern_encoder, batchsize_pattern_encoder, device)
    get_pattern_emb_eval(train_dataset + split_suffix, test_dataset, hidden_size_pattern_encoder, device)

    return run(train_dataset, test_dataset, emb_path(leftList), emb_path(rightList), [dims(leftList), dims(rightList)], [2], [128], [0.0005], 64)

if __name__ == '__main__':
    leftList = ['statistic']
    rightList = ['paragraph', 'word', 'number', 'pattern']
    # rightList = ['paragraph', 'word', 'pattern']
    # rightList = ['paragraph', 'word']
    # rightList = ['word']
    # rightList = ['word', 'pattern']
    # rightList = ['paragraph']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_all('SANTOS_small', 'SANTOS_small', leftList, rightList, device)
    run_all('TUS_small', 'TUS_small', leftList, rightList, device)
    run_all('TUS_large', 'TUS_large', leftList, rightList, device)
    run_all('SANTOS_large', 'SANTOS_large', leftList, rightList, device)
    run_all('SANTOS_small', 'TUS_large', leftList, rightList, device)
    run_all('TUS_small', 'SANTOS_large', leftList, rightList, device)
    # run_all('OpenDataLarge', 'OpenDataLarge', leftList, rightList, device)
