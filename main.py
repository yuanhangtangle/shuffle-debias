from src.model import DebiasModel
from src import data, exp, utils
from torch.nn.utils import clip_grad_norm_
import torch
from collections import defaultdict
CONFIG = 'config.yaml'


def get_recorder(args):
    if args.data == 'mnli':
        header = ['ep', 'loss', 'dev', 'hans']
        types = ['d', 'e', 'f', 'f']
    elif args.data == 'fever':
        header = ['ep', 'loss', 'dev', 'symmetric']
        types = ['d', 'e', 'f', 'f']
    elif args.data in ['qqp', 'qqp-risk']:
        header = ['ep', 'loss', 'dev', 'paw']
        types = ['d', 'e', 'f', 'f']
    else:
        raise Exception('Unknown Dataset')
    return exp.DataLogger(folder=args.folder, header=header, types=types)


CONFIG = './config.yaml'


def main():
    args = exp.load_args(CONFIG)
    logger = utils.EventLogger(args.folder, True)
    if args.debug:
        logger.info('='*30+'DEBUGGING'+'='*30)
    exp.set_seed(args.seed)

    # out_dim: number of classes
    if args.data in ['mnli', 'fever']:
        args.out_dim = 3
    elif args.data in ['qqp']:
        args.out_dim = 2
    else:
        raise Exception('Unknown dataset')

    # build model
    logger.info('initializing bert classifier')
    net = DebiasModel(args).cuda(args.cuda)

    # read data
    train, dev, test = data.get_train_dev_test_set(args.data)
    # facilitate debugging
    if args.debug:
        train = [s[:10*args.batch_size] for s in train]
        dev = [s[:10*args.batch_size] for s in dev]
        test = [s[:10*args.batch_size] for s in test]
    logger.debug('number of training examples: %s' % len(train[0]))
    logger.debug('number of validation examples: %s' % len(dev[0]))
    logger.debug('number of testing examples: %s' % len(test[0]))

    # test iterator
    if args.data == 'mnli':
        train_iter = net.build_data_iterator(*train, shuffle=True, data='mnli')
        dev_iter = net.build_data_iterator(*dev, shuffle=False, data='mnli')
        test_iter = net.build_data_iterator(*test, shuffle=False, data='hans')
        t_iters = [dev_iter, test_iter]
    elif args.data == 'fever':
        train_iter = net.build_data_iterator( *train, shuffle=True, data='fever')
        dev_iter = net.build_data_iterator(*dev, shuffle=False, data='fever')
        test_iter = net.build_data_iterator( *test, shuffle=False, data='fever-symmetric')
        t_iters = [dev_iter, test_iter]
    elif args.data in ['qqp']:
        train_iter = net.build_data_iterator(*train, shuffle=True, data='qqp')
        dev_iter = net.build_data_iterator(*dev, shuffle=False, data='qqp')
        test_iter = net.build_data_iterator(*test, shuffle=False, data='paw')
        t_iters = [dev_iter, test_iter]
    else:
        raise Exception("UNKNOWN dataset")

    # warmup is adopted.
    args.total_steps = len(train_iter) * args.epochs
    args.warmup_steps = int(args.total_steps * args.warmup_proportion)
    # setup optimizer
    optimizer, scheduler = net.setup_optimizers()

    losses = []
    row = []
    logger.info(args.__dict__)
    recorder = get_recorder(args)
    for ep in range(args.epochs):
        print(f'Finetuning at epoch {ep}')
        total_loss = 0
        num = 0
        net.train()
        for loss in train_iter:
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
            total_loss += float(loss)
            num += 1
        net.eval()

        # evaluation
        row = [ep, total_loss/num]
        for it in t_iters:
            _, _, acc = it.infer()
            row.append(acc)
        recorder.print_and_log(row)

    # save the last model
    saver = exp.Saver(net, args, optimizer, scheduler, file_name='checkpoint.pt')
    # note that the first data_iterator is the dev_iter, 
    # thus row[2] is the accuracy in the dev set
    dev_acc = row[2]
    saver.save(dev_acc)


if __name__ == '__main__':
    main()


