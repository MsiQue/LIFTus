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

## Setup

* If you want to run LIFTus in your private data lake, please refer to the following steps:
* Step 1: Similar to the previous operation in *Reproducibility*, build the file tree in the following form:
```
📂 LIFTus
├── 📂 LM
│    ├── 📂 fasttext
│    │    └── wiki-news-300d-1M.vec
│    └── 📂 bert
├── 📂 data_lake
│    ├── [data_lake_name]
│    ├── ...
├── 📂 ground_truth
├── 📂 ... (Other directories can be automatically generated)
```
* Step 2: Create a new folder in the ./data_lake directory, with the folder name being the name of the data lake (denoted as [data_lake_name], which needs to replace the dataset name in the code), and then copy the CSV format table to this folder.
* Step 3: Groundtruth File. It is stored using a pickle file, and the stored object is a dictionary. The keys of the dictionary are the names of the query table, and the values are lists that contain the names of all the tables which can be united with the query table.
* Step 4: To modify the get_ground_truth() function in the *global.py* file, you need to add [data_lake_name] to the actual storage location of the pickle file for ground truth. 
* Step 5: Modify the main function in *run_all.py* to the following:
```python
if __name__ == '__main__':  
    leftList = ['statistic']  
    rightList = ['paragraph', 'word', 'number', 'pattern']  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    run_all('[data_lake_name]', '[data_lake_name]', leftList, rightList, device) 
```
* Step 6:  Run script
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

