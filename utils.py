import os
import math
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import Dataset


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, checkpoint_path, patience=10, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model, epoch)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, epoch)
            self.counter = 0

    def save_checkpoint(self, score, model, epoch):
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
        self.save_epoch = epoch


def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)
    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, num_users, sample_seq


def get_seq_dic(args):
    args.data_file = args.data_dir + args.data_name + '.txt'
    args.sample_file = args.data_dir + args.data_name + '_sample.txt'
    user_seq, max_item, num_users, sample_seq = get_user_seqs_and_sample(args.data_file, args.sample_file)
    seq_dic = {'user_seq_past':user_seq, 'num_users':num_users, 'sample_seq':sample_seq}

    return seq_dic, max_item


def get_dataloder(args, seq_dic):
    train_dataset = Dataset(args, seq_dic['user_seq_past'], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = Dataset(args, seq_dic['user_seq_past'], test_neg_items=seq_dic['sample_seq'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = Dataset(args, seq_dic['user_seq_past'], test_neg_items=seq_dic['sample_seq'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    return train_dataloader, eval_dataloader, test_dataloader, seq_dic['user_seq_past']

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)
