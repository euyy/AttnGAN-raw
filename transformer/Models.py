''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from miscc.config import cfg


__author__ = "Yu-Hsiang Huang"

def words_pooling(words_embs, sum_mask, mode='MEAN'):
    if mode == 'MEAN':
        sum_embeddings = torch.sum(words_embs, 1)
        sum_mask = torch.sum(sum_mask, 1)
        sent_emb = sum_embeddings / sum_mask 
    else:
        raise NotImplementedError()

    return sent_emb

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def get_target_mask(len_s,device):
    ''' For masking out the subsequent info. '''
    # sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=600):
        '''
        description: n_position 指的是位置编码最多可以支持 n_position 个单词，所以 n_position >= max_len，位置编码不参与梯度回传
        param {*} self
        param {*} d_hid: 单词的向量维度
        param {*} n_position: 最多支持 n_position 个单词的编码
        return {*}
        '''
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec=512, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, pad_idx=0, dropout=0.1, n_position=200, scale_emb=False):
        '''
        description: 编码器初始化
        param {*} self
        param {*} n_src_vocab： 单词的词典大小
        param {*} d_word_vec: 单词的映射维度，也就是单词向量的维度
        param {*} n_layers: encoder 有多少层
        param {*} n_head: 多头注意力有几个头
        param {*} d_k: 自注意力中 K 的维度
        param {*} d_v: 自注意力中 V 的维度
        param {*} d_model: 在 encoder layer 之间传递的向量维度
        param {*} d_inner: feedforword 层将 d_model 映射到 d_inner 再映射到 d_model
        param {*} pad_idx: 值为 pad_idx 的数会被映射为 0 ，且不会计算梯度信息
        param {*} dropout: 多处用到，以一定概率丢弃一些向量
        param {*} n_position: 位置编码的个数
        param {*} scale_emb: 是否乘以 sqrt(d_model)
        return {*}
        '''

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        # self.pooling_mode = cfg.TEXT.POOLING_MODE

    def forward(self, src_seq, return_attns=False):
        '''
        description: 
        param {*} self
        param {*} src_seq: [bs, seq_len], 文本序列
        param {*} return_attns: 是否返回 attn map list，会将所有 encoder layer 的attn map 全部返回
        return {*}
        '''
        word_emb_list = []
        src_mask = get_pad_mask(src_seq, pad_idx=0) # ! [bs, 1, max_len]
        enc_slf_attn_list = []

        # -- Forward
        word_emb = self.src_word_emb(src_seq) # ! [bs, max_len, d_word_vec]
        if self.scale_emb:
            word_emb *= self.d_model ** 0.5
        word_emb = self.dropout(self.position_enc(word_emb)) # ! [bs, max_len, d_word_vec]
        word_emb = self.layer_norm(word_emb) # ! [bs, max_len, d_word_vec]

        for enc_layer in self.layer_stack:
            word_emb, enc_slf_attn = enc_layer(word_emb, slf_attn_mask=src_mask)
            word_emb_list.append(word_emb)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        
        attn_mask_expanded = src_mask.squeeze(1).unsqueeze(-1).expand(word_emb.size()).float()
        for word_emb in word_emb_list:
            word_emb = word_emb * attn_mask_expanded
        # sum_mask = src_mask.unsqueeze(-1).sum(1)
        # sent_embs = words_pooling(words_embs = word_emb, sum_mask = sum_mask)
        if return_attns:
            return word_emb, enc_slf_attn_list
        return word_emb,
        # return word_emb_list, sent_embs


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        # self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        # seq_logit = self.trg_word_prj(dec_output)
        # if self.scale_prj:
        #     seq_logit *= self.d_model ** -0.5

        # return seq_logit.view(-1, seq_logit.size(2))

        return dec_output
