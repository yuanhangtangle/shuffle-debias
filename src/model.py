import numpy as np
from abc import abstractmethod
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from transformers import logging as trm_logger
from src.utils import *
from src.loss import *
trm_logger.set_verbosity_error()


class Net(nn.Module):
    ''' base class '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        return None, None

    @abstractmethod
    def setup_optimizers(self, *args, **kwargs):
        return None

    def set_device(self, *tensors):
        device = next(self.parameters()).device
        return set_device(device, *tensors)

    def build_data_iterator(*args, **kwargs):
        pass


class BertEncoder(Net):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = get_tokenizer(args.tokenizer, args.bert_version)
        self.model = get_bert_model(args.bert_version)

    def tokenize(self, sent1, sent2=None):
        assert type(sent1) in (list, tuple) and (
            type(sent2) in (list, tuple) or sent2 is None)
        out = self.tokenizer(
            text=sent1,
            text_pair=sent2,
            padding=True,
            max_length=self.args.max_length,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {
            'input_ids': self.set_device(out['input_ids']),
            'token_type_ids': self.set_device(out['token_type_ids']),
            'attention_mask': self.set_device(out['attention_mask'])
        }
        return inputs

    def pooler_output(self, inputs):
        return self.model(**inputs).pooler_output

    def last_cls(self, inputs):
        '''return: (bs, hidden_dim)'''
        out = self.model(**inputs).last_hidden_state
        cls = out[:, 0, :]
        return cls

    def last_hidden_state(self, inputs):
        '''return: (bs, max_length, hidden_dim)'''
        return self.model(**inputs, output_hidden_states=True).last_hidden_state

    def hidden_states(self, inputs):
        '''return: all hiddent states including word embedding'''
        return self.model(**inputs, output_hidden_states=True).hidden_states

    def get_reps(self, inputs):
        if self.args.rep == 'pooler':
            return self.pooler_output(inputs)
        elif self.args.rep == 'cls':
            return self.last_cls(inputs)
        else:
            raise Exception('Unknown hidden representation type')

    def forward(self, inputs):
        if self.args.rep == 'pooler':
            return self.pooler_output(inputs)
        elif self.args.rep == 'cls':
            return self.last_cls(inputs)
        else:
            raise Exception('Unknown rep `%s`' % self.args.rep)


class FC(nn.Module):
    '''fully connected layer'''

    def __init__(self, in_dim, out_dim, norm, init='gauss0.02', bias=True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.norm = norm
        self.fc = nn.Linear(
            in_features=self.in_dim,
            out_features=self.out_dim, bias=self.bias)
        if init == 'gauss0.02':  # oyyw, magical initialization
            self.fc.weight.data = torch.normal(
                0, 0.02, size=(self.out_dim, self.in_dim))
            if self.bias:
                self.fc.bias.data = torch.normal(
                    0, 0.02, size=(self.out_dim, ))
        elif init == 'xavier':  # following RISK, COLING 2022
            nn.init.xavier_uniform_(self.fc.weight, .1)
            nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        '''
        shape of x: (batch, in_dim)
        '''
        if self.norm:
            norm_x = torch.linalg.norm(x.data, dim=1, keepdim=True)
            x.data /= norm_x
            norm_w = torch.linalg.norm(
                self.fc.weight.data, dim=1, keepdim=True)
            self.fc.weight.data = self.fc.weight.data / norm_w
        return self.fc(x)


class BertCLS(Net):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.max_length = args.max_length
        self.out_dim = args.out_dim
        self.bert = BertEncoder(args)
        self.drop = nn.Dropout(p=args.drop_out)
        self.fc = FC(
            in_dim=hidden_dim(args.bert_version),
            out_dim=args.out_dim,
            norm=False,
            init=args.init)

    def compute_logits(self, pair) -> torch.Tensor:
        out = self.bert.tokenize(pair[0], pair[1])
        rep = self.bert.get_reps(out)
        rep = self.drop(rep)
        cls_logits = self.fc(rep)
        return cls_logits

    def bert_params(self):
        return self.bert.parameters()

    def setup_optimizers(self, lr=None, weight_decay=None, warmup_steps=None, total_steps=None):
        args = self.args
        lr = args.lr if lr is None else lr
        weight_decay = args.weight_decay if weight_decay is None else weight_decay
        warmup_steps = args.warmup_steps if warmup_steps is None else warmup_steps
        total_steps = args.total_steps if total_steps is None else total_steps
        params = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_param = [
            {
                'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        op = AdamW(grouped_param, lr=lr)
        sch = get_scheduler(args.scheduler, op, warmup_steps, total_steps)
        return op, sch

    def infer(self, sent1, sent2, ys=None):
        def collect_func(bs):
            a, b = [], []
            for x, y in bs:
                a.append(x)
                b.append(y)
            return a, b
        loader = DataLoader(
            DatasetBuilder(sent1, sent2),
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=collect_func)
        logits = []
        with torch.no_grad():
            for bs in loader:
                logits.append(self.compute_logits(bs))
        logits = torch.cat(logits, dim=0)
        if ys is None:
            return logits
        else:
            ys = torch.as_tensor(ys, dtype=int).cuda(self.args.cuda)
            acc = (logits.argmax(dim=1) == ys).type(torch.float).mean()
            return logits, ys, float(acc.cpu())


def shuffle_collect_fn(args):
    rng1 = np.random.default_rng(args.seed)
    rng2 = np.random.default_rng(args.seed)

    def func(x):
        sent1, shuffle1 = [], []
        sent2, shuffle2 = [], []
        ys = []
        for t in x:
            s1, s2, y = t
            sent1.append(s1)
            sent2.append(s2)
            ys.append(y)
            for i in range(args.shuffle_times):
                # shuffle the first sentence
                l1 = split_text(s1, args.n_gram)
                rng1.shuffle(l1)
                shuffle1.append(' '.join(l1))
                # shuffle the second sentence
                l2 = split_text(s2, args.n_gram)
                rng2.shuffle(l2)
                shuffle2.append(' '.join(l2))
        orig = (sent1, sent2)
        shuffle = (shuffle1, shuffle2)
        return (orig, shuffle, ys)
    return func


class ShuffleIterator:
    def __init__(self, net,
                 sent1, sent2=None, ys=None,
                 batch_size=1, shuffle=False, data='mnli'):
        self.net = net
        self.labeled = ys is not None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        self.args = self.net.args
        if sent2 is not None:
            self.ds = DatasetBuilder(
                sent1, sent2, ys) if self.labeled else DatasetBuilder(sent1, sent2)
        else:
            self.ds = DatasetBuilder(
                sent1, ys) if self.labeled else DatasetBuilder(sent1)
        self.n = len(self.ds) // self.batch_size

    def __len__(self, ):
        return self.n

    def __iter__(self, ):
        if self.net.training:  # training
            assert self.labeled
            loader = DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                collate_fn=shuffle_collect_fn(self.args))
            for orig, shuffle, ys in tqdm(loader, ncols=40):
                yield self.net(orig=orig, shuffle=shuffle, ys=ys)
        else:  # evaluation
            loader = DataLoader(
                self.ds, batch_size=self.batch_size, shuffle=False)
            for batch in tqdm(loader, ncols=40):
                if self.labeled:  # infer with label
                    s1, s2, ys = batch
                else:  # infer w/o label
                    s1, s2 = batch
                    ys = None
                orig = (s1, s2)
                logits = self.net(orig=orig)
                yield logits, ys

    def infer(self, ):
        self.net.eval()
        logits = []
        ys = []
        with torch.no_grad():
            for l, y in self:
                logits.append(l)
                if self.labeled:
                    ys.append(y)
        logits = torch.cat(logits, dim=0)
        if self.labeled:
            ys = self.net.set_device(torch.cat(ys, dim=0))
            acc = batch_accuracy(logits, ys, data=self.data)
            return logits, ys, acc
        else:
            return logits


class DebiasModel(Net):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = BertCLS(args)
        self.loss = self.get_loss()

    def forward(self, orig, shuffle=None, ys=None):
        cls_logits = self.model.compute_logits(orig)
        if not self.training:
            return cls_logits
        else:
            assert ys is not None
            if type(ys) == list:
                ys = self.set_device(torch.as_tensor(ys, dtype=torch.long))
            if self.args.loss == 'ce':
                loss = self.loss(cls_logits, ys)
            else:
                bs = len(orig[0])
                n = self.args.shuffle_times
                bias_logits = self.model.compute_logits(shuffle)
                bias_probs = torch.softmax(bias_logits, dim=1)
                with torch.no_grad():  # disable BackProp
                    w = bias_probs.detach().view(bs, n, -1).mean(dim=1)
                    if self.args.clip > 0:
                        w = torch.clamp(w, self.args.clip,
                                        1.0 - self.args.clip)
                loss = self.loss(cls_logits, ys, w, inputs_adv_is_prob=True)
            return loss

    def get_loss(self,):
        if self.args.loss == 'ce':
            return nn.CrossEntropyLoss()
        if self.args.loss == 'poe':
            return POELoss(poe_alpha=self.args.poe_alpha)
        elif self.args.loss == 'focal':
            return FocalLoss(alpha=self.args.alpha, gamma=self.args.gamma)
        else:
            raise Exception('Unkown loss type')

    def setup_optimizers(self, ):
        return self.model.setup_optimizers()

    def build_data_iterator(
        self, sent1, sent2, labels=None,
        batch_size=None, shuffle=True, data='mnli'
    ) -> ShuffleIterator:
        if batch_size is None:
            batch_size = self.args.batch_size
        return ShuffleIterator(self, sent1, sent2, labels, batch_size, shuffle, data)

    def freeze_bert(self):
        return freeze(self.model.bert)

    def unfreeze_bert(self):
        return unfreeze(self.model.bert)

    def infer(self, sent1, sent2):
        it = self.build_data_iterator(sent1, sent2)
        return it.infer()
