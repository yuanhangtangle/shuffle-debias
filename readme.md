# Shuffle Debias
This is the github repository for NLPCC 2023 paper **IDOS: A Unified Debiasing Method via Word Shuffling**.

## Data
We use MNLI, FEVER and QQP for training, and HANS, FEVER-symmetric and PAWS for evaluation.

- MNLI & HANS: the MNLI dataset can be downloaded from huggingface. 
    You can load `multi_nli` dataset from HuggingFace and `save_to_disk`.
    `src/data.py` use `datasets.load_from_disk` to load the saved dataset.
    HANS can be downloaded from its official github repository. 
    Download the [ heuristics_train_set.txt ](https://github.com/tommccoy1/hans/blob/master/heuristics_train_set.txt)
    and [ heuristics_evaluation_set.txt ](https://github.com/tommccoy1/hans/blob/master/heuristics_evaluation_set.txt),
    and rename them as `train.txt` and `test.txt`.

- FEVER & FEVER-symmetric: you can download FEVER and FEVER-symmetric datasets using the tools provided by 
    [Rabeehk](https://github.com/rabeehk/robust-nli).

- QQP & PAWS: use the google drive url provided by [RISK](https://github.com/CuteyThyme/RISKA).

For some reasons, I use datasets from different sources. 
You may simply download all the datasets using the google drive url provied by [RISK](https://github.com/CuteyThyme/RISKA),
and write your own parsers to read these datasets. 

## Train & Eval
use `train.sh`. See `config.yaml` for available configurations and explainations.
