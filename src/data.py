from src.utils import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from datasets.load import load_from_disk
from torch.utils.data import Dataset

SEP = 102
PAD = 0
CLS = 101
NLI_LABEL_MAP = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
    'non-entailment': 1
}

FEVER_LABEL_MAP = {
    'SUPPORTS': 0,
    'REFUTES': 1,
    'NOT ENOUGH INFO': 2
}

MNLI = '/home/data_ti6_c/tangyh/data/mnli'
HANS = '/home/data_ti6_c/tangyh/data/hans'
FEVER = '/home/data_ti6_c/tangyh/data/fever'
QQP = '/home/data_ti6_c/tangyh/data/qqp'
PAW = '/home/data_ti6_c/tangyh/data/qqp/paw-qqp'
MNLI_LABEL_MAP = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


def read_mnli(path):
    ds = load_from_disk(path)
    ds.set_format('pandas')
    d = dict()
    for split in ds.keys():
        con = ds[split]['label'] > -1
        d[split] = (
            ds[split]['premise'][con].tolist(),
            ds[split]['hypothesis'][con].tolist(),
            ds[split]['label'][con].tolist()
        )
    d['dev_matched'] = d['validation_matched']
    d['dev_mismatched'] = d['validation_mismatched']
    d.pop('validation_matched')
    d.pop('validation_mismatched')
    return d


def read_fever(path):
    ds = dict()
    for split, name in zip(
        ['train', 'dev', 'test'],
            ['fever.train.jsonl', 'fever.dev.jsonl', 'fever_symmetric_generated.jsonl']):
        claim = []
        evidence = []
        label = []
        p = join_path(path, name)
        with open(p, 'r') as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            d = json.loads(l)
            if split in ['train', 'dev']:
                claim.append(d['claim'])
                evidence.append(d['evidence'])
                label.append(d['gold_label'])
            else:
                claim.append(d['claim'])
                evidence.append(d['evidence_sentence'])
                label.append(d['label'])
        label = [FEVER_LABEL_MAP[i] for i in label]
        ds[split] = (claim, evidence, label)
    return ds


def read_hans(path):
    d = dict()
    for split in ['train', 'test']:
        hans = pd.read_csv(join_path(path, '%s.txt' % split), sep='\t')
        d[split] = (
            hans.sentence1.tolist(),
            hans.sentence2.tolist(),
            hans.gold_label.apply(lambda x: NLI_LABEL_MAP[x]).tolist()
        )
    return d


def read_qqp(path):
    d = dict()
    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(os.path.join(path, '%s.tsv' %
                         split), sep='\t', header=None)
        df.dropna(inplace=True)
        d[split] = (
            df[1].tolist(),
            df[2].tolist(),
            df[0].tolist()
        )
    return d


def read_paw(path):
    d = dict()
    for split in ['train', 'dev_and_test']:
        df = pd.read_csv(os.path.join(path, '%s.tsv' % split), sep='\t')
        df.dropna(inplace=True)
        d[split] = (
            df.sentence1.apply(lambda x: x[2:-1]),
            df.sentence2.apply(lambda x: x[2:-1]),
            df.label.tolist()
        )
    d['test'] = d.pop('dev_and_test')
    return d


def load_dataset(name) -> dict:
    if name == 'snli':
        return read_snli(SNLI)
    elif name == 'mnli':
        return read_mnli(MNLI)
    elif name == 'mnli-easy-hard':
        return read_mnli_easy_hard(MNLI_EASY_HARD)
    elif name == 'hans':
        return read_hans(HANS)
    elif name == 'fever':
        return read_fever(FEVER)
    elif name == 'qqp':
        return read_qqp(QQP)
    elif name == 'qqp-risk':
        return read_qqp_risk(QQP_PAWS_RISK)
    elif name == 'paw':
        return read_paw(PAW)
    elif name == 'paw-risk':
        return read_paw_risk(QQP_PAWS_RISK)
    elif name == 'anli':
        return read_anli_test_set(ANLI)
    elif name == 'clinc14':
        return read_clinc_14_cov(CLINC14)
    else:
        raise Exception('Unknown dataset')


def get_train_dev_test_set(name):
    ds = load_dataset(name)
    if name == 'mnli':
        train = ds['train']
        dev = ds['dev_matched']
        test = load_dataset('hans')['test']
    elif name == 'fever':
        train = ds['train']
        dev = ds['dev']
        test = ds['test']
    elif name == 'qqp':
        train = ds['train']
        dev = ds['dev']
        test = load_dataset('paw')['test']
    elif name == 'qqp-risk':
        train = ds['train']
        dev = ds['dev']
        test = load_dataset('paw-risk')['test']
    elif name == 'clinc14':
        train = ds['train']
        dev = ds['valid_id']
        test = ds['test']
    else:
        raise Exception('Unknown dataset')
    return train, dev, test
