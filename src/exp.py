import csv
import time
import argparse
import os
import torch
import random
import numpy as np
from src.model import *
from src.utils import *
from src.data import *
from typing import Optional, Tuple, Union, Iterable

class Saver:
    def __init__(self, model: torch.nn.Module, args, optimizer, scheduler = None, file_name = None) -> None:
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.file_name = file_name if file_name is not None else 'checkpoint.pt'

    def save(self, score):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        d = {
            'param': model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
            'score': score, 
            'args': self.args,
        }
        torch.save(d, os.path.join(self.args.folder, self.file_name))


class DataLogger:
    '''
    store and print experiemtnal results in csv format
    file_path: file path
    header: header, i.e. column names
    types: format of each column, see `utils.print_row` for more details
        d for decimal, 
        f for float, 
        p for percentage, 
        e for scientific number, 
        s for string
    '''
    def __init__(self, folder=None, header=None, types=None, file_name=None):
        self.file_path = os.path.join(folder, 'data.csv' if file_name is None else file_name)
        self.header = header
        self.types = types if types is not None else ['s'] * len(header)
        assert len(self.header) == len(self.types)
        
        self.f = None
        self.writer = None
        if self.file_path is not None:
            self.f = open(self.file_path, 'w', newline='')
            self.writer = csv.writer(self.f, header)
            self.writer.writerow(header)

    def log(self, data: Iterable):
        assert self.writer is not None
        self.writer.writerow(data)
        self.f.flush()

    def print_header(self):
        print_header(self.header)

    def print_row(self, data):
        print_row(data, self.types)
  
    def print_and_log(self, data):
        self.print_header()
        self.print_row(data)
        self.log(data)

def load_args(path, check_exists=False):
    cft = load_yaml(path)
    parser = argparse.ArgumentParser()
    cft.time = time.strftime("%Y-%m-%d-%H-%M-%S")   
    for key, value in cft.__dict__.items():
        parser.add_argument('--{}'.format(key), type=type(value), default=value)
    args = parser.parse_args()
    args.folder = os.path.join(args.base_folder, "seed%s" % args.seed)
    if not args.debug and check_exists:
        assert not os.path.exists(args.folder)
        os.makedirs(args.folder)
    elif not os.path.exists(args.folder):
        os.makedirs(args.folder)

    save_yaml(args.__dict__, os.path.join(args.folder, 'config.yaml'))
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False

def restore_checkpoint(folder=None, args=None, file_name = None, model=None) -> Tuple[Net, Container]:
    if folder is None:
        assert args is not None
        folder = os.path.join(args.base_folder, 'seed%s'%args.seed)
    if file_name is None:
        file_name = 'checkpoint.pt'
    d = torch.load(os.path.join(folder, file_name), map_location='cpu')
    model_args = load_yaml(os.path.join(folder, 'config.yaml'))
    net = DebiasModel(model_args)
    net.load_state_dict(d['param'])
    return net, model_args

