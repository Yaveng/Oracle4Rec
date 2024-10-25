import tqdm
import torch
import numpy as np
from models import Oracle4Rec
from torch.optim import Adam
from utils import get_metric


class Trainer:
    def __init__(self, train_dataloader, eval_dataloader, test_dataloader, user_seq, args):
        self.num_item = args.item_size
        self.args = args
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(args.cudaid) if self.cuda_condition else "cpu")
        self.model = Oracle4Rec(args=args, device=self.device).to(self.device)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim_future = Adam(self.model.future_ae.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.optim = Adam(list(self.model.past_ae.parameters())+list(self.model.transition.parameters()), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

    def train(self, epoch):
        rec_data_iter = tqdm.tqdm(enumerate(self.train_dataloader), ncols=120, desc='Train in {}-th epoch'.format(epoch), total=len(self.train_dataloader))
        self.model.train()
        rec_loss = 0.0
        for i, batch in rec_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            _, input_ids, answer, neg_answer, input_ids_future, answer_future, neg_answer_future = batch

            # Share embeddings between two encoders
            self.model.future_ae.item_embeddings.weight.data = self.model.past_ae.item_embeddings.weight.data.detach()
            self.model.future_ae.position_embeddings.weight.data = self.model.past_ae.position_embeddings.weight.data.detach()

            # 2PTraining-Train future encoder
            future_loss, _, _ = self.model.future_forward(input_ids_future, answer_future, neg_answer_future)
            self.optim_future.zero_grad()
            future_loss.backward()
            self.optim_future.step()
            rec_loss += future_loss.item()
            # Obtain future information
            _, z_future, z_future_mask = self.model.future_forward(input_ids_future, answer_future, neg_answer_future)
            z_future = z_future.detach()

            # 2PTraining-Train past encoder
            self.model.past_ae.item_embeddings.weight.data = self.model.future_ae.item_embeddings.weight.data.detach()
            self.model.past_ae.position_embeddings.weight.data = self.model.future_ae.position_embeddings.weight.data.detach()
            loss, z_past = self.model.past_forward(input_ids, answer, neg_answer)
            rec_loss += loss.item()

            # Achieve oracle guiding
            transition_loss = self.model.transition_forward(z_past, z_future, z_future_mask)
            loss += self.args.alpha * transition_loss
            rec_loss += self.args.alpha * transition_loss.item()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            post_fix = {'rec_loss': '{:.4f}'.format(rec_loss / len(rec_data_iter))}
            rec_data_iter.set_postfix(post_fix)

    def valid_test(self, epoch, is_valid=True):
        if is_valid:
            phase = 'Valid'
            dataloader = self.eval_dataloader
        else:
            phase = 'Test'
            dataloader = self.test_dataloader

        rec_data_iter = tqdm.tqdm(enumerate(dataloader), ncols=120, desc='{} in {}-th epoch'.format(phase, epoch), total=len(dataloader))

        self.model.eval()
        pred_list = None
        for i, batch in rec_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, answers, _, sample_negs = batch
            test_neg_items = torch.cat((answers.unsqueeze(-1), sample_negs), -1)
            test_logits = self.model.predict(input_ids, test_neg_items)
            test_logits = test_logits.cpu().detach().numpy().copy()
            if i == 0:
                pred_list = test_logits
            else:
                pred_list = np.append(pred_list, test_logits, axis=0)
        res, res_str = self.get_scores(pred_list)
        return res, res_str

    def get_scores(self, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        res_str = 'HR@1:{:.4f}\tHR@5:{:.4f}\tNDCG@5:{:.4f}\tHR@10:{:.4f}\tNDCG@10:{:.4f}\tMRR:{:.4f}'.format(
            HIT_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR,
        )
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], res_str

