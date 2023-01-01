from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F
import math
from einops import repeat
from torch.autograd import Variable
 
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SingleTransformer(nn.Module):
    def __init__(self, feat_in, hidden=768, nheads=8, nlayers=6, dropout=0.1):
        super().__init__()
        # from transformer import TransformerEncoderLayer
        self.feat_in = feat_in
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_in))#用这个token来聚合特征
        self.pos_encoder = PositionalEncoding(feat_in, dropout) #根据feat_in生成位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_in, nhead=nheads, dropout=dropout)
        # encoder_layer = TransformerEncoderLayer(d_model=feat_in, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers, norm=nn.LayerNorm(feat_in))
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_in, hidden))

    def forward(self, feats, output_all=False, key_padding_mask=None):
        # input: bs, seq, feat_in; output: bs, hidden
        b, n, _ = feats.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        feats = torch.cat((cls_tokens, feats), dim=1).permute(1, 0, 2)
        feats = self.pos_encoder(feats)
        if key_padding_mask is not None:
            cls_mask = key_padding_mask.new_zeros(b, 1)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
            feats = self.transformer_encoder(feats, src_key_padding_mask=key_padding_mask)
        else:
            feats = self.transformer_encoder(feats)
        # feats = self.transformer_encoder(feats)
        if output_all:
            return self.mlp(feats.permute(1, 0, 2))
        return self.mlp(feats[0])


class SingleTransformerClassifier(nn.Module):
    def __init__(self, feat_in, hidden=768, nheads=8, nlayers=6, dropout=0.1, num_classes=2513):
        super().__init__()
        self.single_transformer = SingleTransformer(feat_in, hidden, nheads, nlayers, dropout)
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, feats, output_feats=False):
        if output_feats:
            return self.single_transformer(feats)
        return self.linear(self.single_transformer(feats))


class SingleTransformerPretrain(nn.Module):
    def __init__(self, feat_in, hidden=768, nheads=8, nlayers=6, dropout=0.1):
        super().__init__()
        # self.single_transformer = SingleTransformer(feat_in, hidden, nheads, nlayers, dropout)
        self.transformer_encoder = SingleTransformer(feat_in, hidden, nheads, nlayers, dropout)
        self.linear = nn.Linear(hidden, feat_in)

    def forward(self, feats, output_all=False):
        # input: bs, seq, feat_in; output: bs, feat_in
        # return self.linear(self.single_transformer(feats, output_all=output_all))
        return self.linear(self.transformer_encoder(feats, output_all=output_all))


class FusionTransformer(nn.Module):
    def __init__(self, modality_in, hidden=768, feat_out=1024, num_class=2513, nheads=8, nlayers=6, dropout=0.1):
        super(FusionTransformer, self).__init__()
        self.pre_cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.fut_cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.modality_embeddings = nn.Parameter(torch.randn(1, modality_in, hidden))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers, norm=nn.LayerNorm(hidden))
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, feat_out))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_out, num_class))

    def forward(self, modality_feats, task):
        b, n, _ = modality_feats.shape
        modality_feats += self.modality_embeddings[:, :(n + 1)]
        pre_cls_tokens = repeat(self.pre_cls_token, '() n d -> b n d', b=b)
        fut_cls_tokens = repeat(self.fut_cls_token, '() n d -> b n d', b=b)
        modality_feats = torch.cat((fut_cls_tokens, modality_feats), dim=1)
        modality_feats = torch.cat((pre_cls_tokens, modality_feats), dim=1).permute(1, 0, 2)
        modality_feats = self.transformer_encoder(modality_feats)
        hpre = self.mlp_head(modality_feats[0])
        hfut = self.mlp_head(modality_feats[1])
        if task == 'pre':
            action = self.fc(hpre)
        else:
            action = self.fc(hfut)
        return action, hpre, hfut

class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        device = x.device
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            mask = ~mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(3)
            attention = attention.masked_fill(mask, -float("inf"))  # batch_size, L, L, head
        zeros = torch.zeros(attention.shape[2], attention.shape[2]).to(device) # 256,256
        mask_templates = torch.eye(4).to(device)
        zeros[:4,:4] = mask_templates
        mask_templates = zeros.bool()
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class FusionLinear(nn.Module):
    def __init__(self, modality_in, hidden=768, feat_out=1024, num_class=2513, nheads=8, nlayers=6, dropout=0.5):
        super(FusionLinear, self).__init__()
        self.num_class = num_class
        self.feat_out = feat_out
        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * modality_in, self.feat_out * 2),
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_out, self.num_class)
        )
    def forward(self, modality_feats, task):
        b, n, d = modality_feats.shape
        modality_feats = self.mlp_head(modality_feats.view(b, n * d))
        hpre = modality_feats[:, :self.feat_out]
        hfut = modality_feats[:, self.feat_out:]
        if task == 'pre':
            action = self.fc(hpre)
        else:
            action = self.fc(hfut)
        return action, hpre, hfut
