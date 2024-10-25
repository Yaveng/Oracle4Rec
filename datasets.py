import torch
import random
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq_past = []
        self.user_seq_future = []
        self.user_items = []
        self.max_len = args.max_seq_length
        self.future_max_len = args.future_max_seq_length
        self.past_max_len = self.max_len - self.future_max_len

        if data_type=='train':
            for seq in user_seq:
                input_ids = seq[-(self.max_len + 2):-2] 
                for i in range(len(input_ids)):
                    self.user_seq_past.append(input_ids[:i + 1])  # [0, 1, ..., i-1, i (answer)]
                    self.user_seq_future.append(input_ids[i:])    # [i, i+1, i+2, ...]
                    self.user_items.append(input_ids)
        elif data_type=='valid':
            for sequence in user_seq:
                self.user_seq_past.append(sequence[:-1])
                self.user_items.append(sequence[:-1])
        else:
            self.user_seq_past = user_seq
            self.user_items = user_seq

        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __len__(self):
        return len(self.user_seq_past)

    def __getitem__(self, index):
        seq_set = set(self.user_items[index])
        past_items = self.user_seq_past[index]

        input_seq_past = past_items[:-1]   # [1, 2, ..., i-1]
        ans_seq_past = past_items[-1]      # [i]
        neg_ans_seq_past = neg_sample(seq_set, self.args.item_size)
        pad_len = self.max_len - len(input_seq_past)
        input_seq_past = [0] * pad_len + input_seq_past
        input_seq_past = input_seq_past[-self.max_len:]
        assert len(input_seq_past) == self.max_len

        if self.data_type == 'train':
            future_items = self.user_seq_future[index]
            if len(future_items) < 2:
                input_seq_future = future_items
                ans_seq_future = [0]
            else:
                input_seq_future = future_items[:-1]
                ans_seq_future = future_items[1:]
            input_seq_future = input_seq_future[:self.future_max_len]
            ans_seq_future = ans_seq_future[:self.future_max_len]

            pad_len_input = self.future_max_len - len(input_seq_future)
            # [i, i+1, ..., i+future_max_len-1]
            input_seq_future = input_seq_future + [0] * pad_len_input
            # [i-past_max_len, ..., i-1] + [i, i+1, ..., i+future_max_len-1]
            # -> [i-past_max_len, ..., i-1, i, i+1, ..., i+future_max_len-1]
            input_seq_future = input_seq_past[-self.past_max_len:] + input_seq_future

            pad_len_ans = self.future_max_len - len(ans_seq_future)
            # [i+1, i+2, i+3, ..., i+future_max_len]
            ans_seq_future = ans_seq_future + [0] * pad_len_ans
            # [i-past_max_len+1, ..., i-1] + [i] + [i+1, ..., i+future_max_len]
            # -> [i-past_max_len+1, ..., i-1, i, i+1, ..., i+future_max_len]
            ans_seq_future = input_seq_past[-self.past_max_len+1:] + [ans_seq_past] + ans_seq_future

            neg_ans_seq_future = []
            for idx in range(len(ans_seq_future)):
                neg_ans_seq_future.append(neg_sample(seq_set, self.args.item_size))

            cur_tensors = (
                torch.tensor(index, dtype=torch.long),
                torch.tensor(input_seq_past, dtype=torch.long),
                torch.tensor(ans_seq_past, dtype=torch.long),
                torch.tensor(neg_ans_seq_past, dtype=torch.long),
                torch.tensor(input_seq_future, dtype=torch.long),
                torch.tensor(ans_seq_future, dtype=torch.long),
                torch.tensor(neg_ans_seq_future, dtype=torch.long)
            )
        else: 
            test_samples = self.test_neg_items[index]
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),
                torch.tensor(input_seq_past, dtype=torch.long),
                torch.tensor(ans_seq_past, dtype=torch.long),
                torch.tensor(neg_ans_seq_past, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item