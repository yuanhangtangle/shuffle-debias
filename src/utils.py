import json
import logging
import os
import pickle as pkl

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel, BertTokenizer
from transformers.optimization import (
    get_constant_schedule, get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup)

SEP = 102
PAD = 0
CLS = 101


class Container:
    def __init__(self, d=None) -> None:
        if d is None:
            return
        if not type(d) == dict:
            d = d.__dict__
        for k, v in d.items():
            self.__dict__[k] = v

    def keys(self):
        return self.__dict__.keys()

    def update(self, args=None, p=None):
        assert args is not None or p is not None
        assert not (args is not None and p is not None)
        if p is not None:
            args = load_yaml(p)
        for k, v in args.__dict__.items():
            if k not in self.keys():
                self.__dict__[k] = v


class EventLogger:
    def __init__(self, folder, debug: bool = False):
        self.init_logger(folder, debug)
        self._logger = logging.getLogger('logger')

    def info(self, s):
        self._logger.info(s)

    def debug(self, s):
        self._logger.debug(s)

    def init_logger(self, folder, debug: bool = False):
        fmt = logging.Formatter(
            '%(asctime)s %(levelname)-5s %(filename)-8s:%(lineno)-3s  %(message)s')

        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG if debug else logging.INFO)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(os.path.join(folder, 'event.log'), 'w')
        fh.setFormatter(fmt)
        logger.addHandler(fh)


class DatasetBuilder(Dataset):
    def __init__(self, *args):
        self.inputs = args

    def __len__(self):
        return len(self.inputs[-1])

    def __getitem__(self, item):
        return tuple(input[item] for input in self.inputs)


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def save_pickle(p, obj):
    with open(p, 'wb') as f:
        pkl.dump(obj, f)


def load_pickle(p):
    with open(p, 'rb') as f:
        return pkl.load(f)


def save_yaml(obj, p):
    with open(p, 'w') as f:
        f.write(yaml.dump(obj))


def load_yaml(p):
    with open(p, 'r') as f:
        r = Container(yaml.load(f, yaml.FullLoader))
        return r


def save_json(obj, p):
    with open(p, 'w') as f:
        json.dump(obj, f)


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def join_path(*args):
    return os.path.join(*args)


def print_header(header):
    print(''.join([n.ljust(10) for n in header]))


def print_row(row, types):
    """
    types: 
        d for decimal, 
        f for float, 
        p for percentage, 
        e for scientific, 
        s for string
    """
    line = []
    for v, t in zip(row, types):
        s = ''
        if t == 'f':
            s = f'{v:.3f}'
        elif t == 'p':
            s = f'{v * 100:.2f}%'
        elif t == 'e':
            s = f'{v:.2e}'
        else:
            s = str(v)
        line.append(s.ljust(10))
    print(''.join(line))


def print_header_row(header, row, types):
    print_header(header)
    print_row(row, types)


def exist_path(p):
    return os.path.exists(p)


def batch_accuracy(preds, ys, data=None):
    n = len(preds.shape)
    if n == 1:  # index
        preds = preds.cpu()
        ys = ys.cpu()
        # if hans
        if data == 'hans':  # hans
            print('=' * 30, 'BE SURE that you are evaluating on HANS dataset', '=' * 30)
            preds = preds.map_(preds, lambda x, y: 0 if x == 0 else 1)
        acc = (preds == ys).sum().item() / len(ys)
        return acc
    elif n == 2:  # (bs, logits)
        preds = torch.argmax(preds, dim=1)
        return batch_accuracy(preds, ys, data)
    elif n == 3:  # (model, bs, logits)
        accs = []
        preds = []
        for p in preds:
            p, a = batch_accuracy(p, ys, data)
            preds.append(p)
            accs.append(a)
        return preds, accs
    else:
        raise Exception('Invalid input shape %s' % n)


def split_text(text, n):
    if text.strip() == '':
        return ''
    assert n > 0
    result = []
    l = text.split()
    t = ''
    for i, c in enumerate(l):
        t = t + ' ' + c
        if (i + 1) % n == 0:
            result.append(t.strip())
            t = ''
    result.append(t.strip())
    if result[0] == "":
        result.pop(0)
    if result[-1] == "":
        result.pop(-1)
    return result


def set_device(cuda, *tensors):
    if len(tensors) == 1:
        return tensors[0].cuda(cuda)
    else:
        return [t.cuda(cuda) for t in tensors]


def get_tokenizer(type, version):
    if type == 'bert':
        return BertTokenizer.from_pretrained(version, local_files_only=True)
    elif type == 'auto':
        return AutoTokenizer.from_pretrained(version, local_files_only=True)
    else:
        raise Exception('Unknown tokenizer type')


def get_bert_model(version):
    if 'bert-base-uncased' in version:
        return BertModel.from_pretrained(version, local_files_only=True)
    else:
        raise Exception('Unknown tokenizer type')


def get_scheduler(scheduler: str, optimizer, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    if scheduler == 'constantlr':
        return get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


def hidden_layers(bert_version):
    if bert_version == 'bert-base-uncased':
        return 13  # embedding + 12 hidden layers
    else:
        raise Exception('unknown encoder version')


def hidden_dim(bert_version):
    if 'bert-base-uncased' in bert_version:
        return 768
    else:
        raise Exception('unknown encoder version')
