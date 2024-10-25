import os
import torch
import argparse
import numpy as np
import datetime
from trainers import Trainer
from utils import EarlyStopping, set_seed, get_seq_dic, get_dataloder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="ml1m", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--cudaid", default="0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--postfix", default='', type=str)
    parser.add_argument("--model_name", default="oracle4rec", type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--future_max_seq_length", default=10, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)

    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_filter_layers", default=1, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)

    parser.add_argument("--hidden_act", default="gelu", type=str)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_hidden_layers", default=5, type=int)
    parser.add_argument("--decay_factor", default=0.2, type=float)
    parser.add_argument("--ratio", default=0.75, type=float)

    args = parser.parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir+'model/'):
        os.makedirs(args.output_dir+'model/')

    seq_dic, max_item = get_seq_dic(args)
    args.item_size = max_item + 1

    args.checkpoint_path = os.path.join(args.output_dir+'model.pt')

    # Load data
    train_dataloader, eval_dataloader, test_dataloader, user_seq = get_dataloder(args,seq_dic)

    # Initialize trainer
    trainer = Trainer(train_dataloader, eval_dataloader, test_dataloader, user_seq, args)
    early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)

    # Train model and validate model
    for epoch in range(args.epochs):
        trainer.train(epoch)
        scores, _ = trainer.valid_test(epoch, is_valid=True)
        early_stopping(np.array(scores[-1:]), trainer.model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Test model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.valid_test(0, is_valid=False)
    print('Test results: {}'.format(result_info))


if __name__=='__main__':
    main()
