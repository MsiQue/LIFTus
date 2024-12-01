# LIFTus: An Adaptive Multi-aspect Column Representation Learning for Table Union Search

This repository is the implementation of paper LIFTus: An Adaptive Multi-aspect Column Representation Learning for Table Union Search.

## Abstract
Table union search (TUS) represents a fundamental operation in data lakes to find tables unionable to the given one.
Recent approaches to TUS mainly learn column representations for searching by introducing Pre-trained Language Models (PLMs) effectively, especially on columns with linguistic data.
However, a significant amount of non-linguistic data, notably represented by domain-specific strings and numerical data in the data lake, are still under-explored in the existing methods.
To address this issue, we propose LIFTus, an adaptive multi-aspect column representation for table unionable search, where aspect refers to a concept more flexible than data types, so that a single column can exhibit multiple aspects simultaneously.
LIFTus aims at combining different aspects of a column (including both linguistic and non-linguistic aspects) to promote the effectiveness and generalization of TUS in a self-supervised manner.
Specifically, besides employing PLMs to extract the linguistic aspects from an individual column, LIFTus trains a pattern encoder to learn possible character-level sequential patterns for the column, and builds a number encoder to capture numerical aspects of the column, including the distribution and magnitude features.
LIFTus further utilizes a hierarchical cross-attention aided by aspect-relevant statistics to combine these aspects adaptively in producing the final column representations, which are indexed by vector retrieval techniques to achieve efficient search.
Extensive experimental results demonstrate that LIFTus has outperformed the current state-of-the-art methods in terms of effectiveness, and achieved much better generalization capability to support unseen data.

## Dataset
To evaluate LIFTus, we use 4 datasets that are widely used in the TUS task in previous works.

* **TUS small** and **TUS large** dataset: please visit [this link](https://github.com/RJMillerLab/table-union-search-benchmark).

* **SANTOS small** and **SANTOS large** dataset: please visit [this link](https://zenodo.org/records/7758091).

## Reproducibility

* Download the code

```
git clone https://github.com/MsiQue/LIFTus.git
cd LIFTus
```
* Since we adopt fastTest as word encoder of LIFTus, please download *wiki-news-300d-1M.vec* from [this link](https://fasttext.cc/docs/en/english-vectors.html). After unzipping it, move it to the directory ./LM/fasttext.

* Move all the tables within each of the four datasets respectively to the directory ./data_lake/[dataset_name]. Then move groundtruth file to ./ground_truth
```
📂 LIFTus
├── 📂 LM
│    ├── 📂 fasttext
│    │    └── wiki-news-300d-1M.vec
│    └── 📂 bert
├── 📂 data_lake
│    ├── SANTOS_small
│    ├── SANTOS_large
│    ├── TUS_small
│    └── TUS_large 
├── 📂 ground_truth
├── 📂 ... (Other directories can be automatically generated)
```
* Run script
```
python run_all.py
```

## Requirements
* python 3.7.10
* pytorch 1.8.1
* transformers 4.16.2
* faiss 1.7.2
* nltk 3.6.7

## Contact
If you have any questions about the paper and the code, please contact qem@stu.pku.edu.cn.

