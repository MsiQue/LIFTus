# LIFTus: An Adaptive Multi-aspect Column Representation Learning for Table Union Search

## Step
*Download the code
```
git clone https://github.com/MsiQue/LIFTus.git
cd LIFTus
```
*Download and copy the necessary files to ./LM (pagagraph encoder and word encoder)
```
paragraph encoder: bert_config.json  config.json  pytorch_model.bin  vocab.txt -> ./LM/bert
word encoder: wiki-news-300d-1M.vec -> ./LM/fast/text
```
*Copy data lake to ./data_lake

*Run script
```
python run_all.py
```

## Requirements
* Python 3.7.10
* PyTorch 1.8.1
* Transformers 4.9.2

## Contact
If you have any questions about the paper and the code, please contact qem@stu.pku.edu.cn.

