import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FilterLayer(nn.Module):
    def __init__(self, args, device):
        super(FilterLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.ratio = args.ratio
        self.device = device

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.abs(torch.fft.rfftfreq(input_tensor.shape[-2])).to(self.device)
        select_weight = weight < weight.quantile(q=self.ratio).item()
        ones = torch.ones(hidden).to(self.device) > 0
        select_weight = torch.outer(select_weight, ones)
        select_weight = torch.unsqueeze(select_weight, dim=0)
        x = x * select_weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act
        self.dense_2 = nn.Linear(4 * args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, args, device):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output


class MainModule(nn.Module):
    def __init__(self, args, device):
        super(MainModule, self).__init__()
        filterlayer = FilterLayer(args, device)
        self.filter_layer = nn.ModuleList([copy.deepcopy(filterlayer) for _ in range(args.num_filter_layers)])
        layer = Layer(args, device)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for filter_layer_module in self.filter_layer:
            hidden_states = filter_layer_module(hidden_states)
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Encoder(nn.Module):
    def __init__(self, args, device):
        super(Encoder, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0).to(device)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size).to(device)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = MainModule(args, device)
        self.future_attn_range = args.future_max_seq_length
        self.device = device

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    def forward(self, input_ids, pos_ids, neg_ids, mode='past'):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.to(self.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Obtain item embedding
        sequence_emb = self.add_position_embedding(input_ids)
        # Obtain item embedding with sequential information encoded
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask)
        seq_out = item_encoded_layers[-1]

        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)

        if mode == 'past':
            seq_emb = seq_out[:, -1, :]
            pos_logits = torch.sum(pos_emb * seq_emb, -1)
            neg_logits = torch.sum(neg_emb * seq_emb, -1)
            loss = torch.mean(- torch.log(torch.sigmoid(pos_logits) + 1e-24) - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24))
            return loss, seq_emb
        else:
            seq_emb = seq_out
            pos_logits = torch.sum(pos_emb * seq_emb, -1)
            neg_logits = torch.sum(neg_emb * seq_emb, -1)
            non_pad_mask = pos_ids != 0
            loss = torch.mean(- torch.log(torch.sigmoid(pos_logits[non_pad_mask]) + 1e-24) - torch.log(1 - torch.sigmoid(neg_logits[non_pad_mask]) + 1e-24))
            seq_emb_partial = seq_emb[:, -self.future_attn_range - 1:, :]
            non_pad_mask = non_pad_mask[:, -self.future_attn_range - 1:]
            return loss, seq_emb_partial, non_pad_mask.detach()

    def predict(self, input_ids, test_neg_sample):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.to(self.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask)
        seq_out = item_encoded_layers[-1]
        seq_out = seq_out[:,-1,:]

        test_item_emb = self.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits


class Transition(nn.Module):
    def __init__(self, seq_length, decay_factor, device='cpu'):
        super(Transition, self).__init__()
        decay_vector = torch.FloatTensor([[i] for i in range(seq_length)]).to(device)
        self.decay_vector = torch.exp(-1.0 * decay_factor * decay_vector)
        self.seq_length = seq_length
        self.to(device)

    def forward(self, z_past, z_future, z_future_mask):
        z_past = torch.repeat_interleave(torch.unsqueeze(z_past, dim=1), repeats=self.seq_length, dim=1)
        z_res = nn.functional.kl_div(z_past.softmax(dim=-1).log(), z_future.softmax(dim=-1), reduction='none')
        z_res = z_res * torch.unsqueeze(z_future_mask, dim=-1)
        # Attenuation of discrepancies between past information and future information
        z_res = torch.einsum('ijk, jl->ijk', z_res, self.decay_vector)
        loss = torch.sum(z_res)
        return loss


class Oracle4Rec(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Oracle4Rec, self).__init__()
        self.item_num = args.item_size
        self.emb_dim = args.hidden_size
        self.device = device

        self.past_ae = Encoder(args, device)
        self.future_ae = Encoder(args, device)
        self.transition = Transition(args.future_max_seq_length+1, args.decay_factor, device)

        self.to(device)

    def past_forward(self, attn_seq, pos_ids, neg_ids):
        loss, z_past = self.past_ae(attn_seq, pos_ids, neg_ids, mode='past')
        return loss, z_past

    def future_forward(self, attn_seq, pos_ids, neg_ids):
        loss, z_future, z_future_mask = self.future_ae(attn_seq, pos_ids, neg_ids, mode='future')
        return loss, z_future, z_future_mask

    def transition_forward(self, z_past, z_future, z_future_mask):
        loss = self.transition(z_past, z_future, z_future_mask)
        return loss

    def predict(self, attn_seq, neg_seq):
        test_logits = self.past_ae.predict(attn_seq, neg_seq)
        return test_logits
