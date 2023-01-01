from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F
import math
from einops import repeat


class OpenLSTM(nn.Module):
    """"An LSTM implementation that returns the intermediate hidden and cell states.
    The original implementation of PyTorch only returns the last cell vector.
    For RULSTM, we want all cell vectors computed at intermediate steps"""

    def __init__(self, feat_in, feat_out, num_layers=1, dropout=0):
        """
            feat_in: input feature size
            feat_out: output feature size
            num_layers: number of layers
            dropout: dropout probability
        """
        super(OpenLSTM, self).__init__()

        # simply create an LSTM with the given parameters
        self.lstm = nn.LSTM(feat_in, feat_out, num_layers=num_layers, dropout=dropout)

    def forward(self, seq):
        # manually iterate over each input to save the individual cell vectors
        last_cell = None
        last_hid = None
        hid = []
        cell = []
        for i in range(seq.shape[0]):
            el = seq[i, ...].unsqueeze(0)
            if last_cell is not None:
                _, (last_hid, last_cell) = self.lstm(el, (last_hid, last_cell))
            else:
                _, (last_hid, last_cell) = self.lstm(el)
            hid.append(last_hid)
            cell.append(last_cell)

        return torch.stack(hid, 0), torch.stack(cell, 0)


class RULSTM(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1,
                 sequence_completion=False, return_context=False):

        super(RULSTM, self).__init__()
        self.feat_in = feat_in
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden
        self.rolling_lstm = OpenLSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth > 1 else 0)
        self.unrolling_lstm = nn.LSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth > 1 else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_class))

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        x, c = self.rolling_lstm(self.dropout(inputs))
        x = x.contiguous()  # batchsize x timesteps x hidden
        c = c.contiguous()  # batchsize x timesteps x hidden
        predictions = []  # accumulate the predictions in a list
        for t in range(x.shape[0]):
            # get the hidden and cell states at current time-step
            hid = x[t, ...]
            cel = c[t, ...]
            ins = inputs[t:, ...]
            h_t, (_, _) = self.unrolling_lstm(self.dropout(ins), (hid.contiguous(), cel.contiguous()))
            h_n = h_t[-1, ...]
            predictions.append(h_n)
        x = torch.stack(predictions, 1)
        return x


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

    def forward(self, x, pe_start=0):
        x = x + self.pe[pe_start:pe_start+x.size(0)]
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

    def forward(self, feats, output_all=False, key_padding_mask=None, pe_start=0):
        # input: bs, seq, feat_in; output: bs, hidden
        b, n, _ = feats.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        feats = torch.cat((cls_tokens, feats), dim=1).permute(1, 0, 2)
        feats = self.pos_encoder(feats, pe_start=pe_start)
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


class SingleTransformerPretrainCTXRec(nn.Module):
    def __init__(self, feat_in, hidden=768, num_class=2513, nheads=8, nlayers=6, dropout=0.1):
        super().__init__()
        # self.single_transformer = SingleTransformer(feat_in, hidden, nheads, nlayers, dropout)
        self.transformer_encoder = SingleTransformer(feat_in, hidden, nheads, nlayers, dropout)
        self.classifier = nn.Linear(hidden, num_class)
        self.linear = nn.Linear(hidden, feat_in)

    def forward(self, feats, output_all=False):
        # input: bs, seq, feat_in; output: bs, feat_in
        # return self.linear(self.single_transformer(feats, output_all=output_all))
        feats = self.transformer_encoder(feats, output_all=output_all)
        action = self.classifier(feats)
        feats_rec = self.linear(feats)
        return action, feats_rec


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


class FusionTransformerFull(nn.Module):
    def __init__(self, modality_in=4, hidden=768, feat_out=1024, num_class=2513, nheads=8, nlayers=1, dropout=0.1):
        super(FusionTransformerFull, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.modality_embeddings = nn.Parameter(torch.randn(1, modality_in, hidden))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers, norm=nn.LayerNorm(hidden))
        self.mlp_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, feat_out))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_out, num_class))

    def forward(self, feats0, feats1, feats2, feats3):
        b, n, _ = feats1.shape
        feats0 += self.modality_embeddings[:, [0]]
        feats1 += self.modality_embeddings[:, [1]]
        feats2 += self.modality_embeddings[:, [2]]
        feats3 += self.modality_embeddings[:, [3]]
        # modality_feats += self.modality_embeddings[:, :n + 1)]
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        modality_feats = torch.cat((cls_token, feats0, feats1, feats2, feats3), dim=1).permute(1, 0, 2)
        modality_feats = self.transformer_encoder(modality_feats)
        hidden = self.mlp_head(modality_feats[0])
        action = self.fc(hidden)
        return action, hidden


class FusionLinear(nn.Module):
    def __init__(self, modality_in, hidden=768, feat_out=1024, num_class=2513, nheads=8, nlayers=6, dropout=0.5):
        super(FusionLinear, self).__init__()
        self.num_class = num_class
        self.feat_out = feat_out
        self.mlp_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, self.feat_out * 2),
        )
        # self.mlp_head = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden, self.feat_out * 2),
        # )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_out, self.num_class)
        )

        # self.fc = nn.Sequential(
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(feat_out, feat_out),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(feat_out, self.num_class),
        # )

    def forward(self, modality_feats, task):
        b, n, d = modality_feats.shape
        # modality_feats = self.mlp_head(modality_feats[:, 2])
        modality_feats = self.mlp_head(modality_feats.view(b, n * d))
        # modality_feats = self.mlp_head(modality_feats.max(1)[0])
        # modality_feats = self.mlp_head(torch.prod(modality_feats, 1))
        hpre = modality_feats[:, :self.feat_out]
        hfut = modality_feats[:, self.feat_out:]
        # hpre = F.normalize(hpre, p=2, dim=-1) * self.feat_out ** 0.5
        # hfut = F.normalize(hfut, p=2, dim=-1) * self.feat_out ** 0.5
        #print("final shape:", hpre.shape)
        # actions = self.fc(modality_feats.view(b, n * d))
        if task == 'pre':
            action = self.fc(hpre)
        else:
            action = self.fc(hfut)
        return action, hpre, hfut


class SingleTransformer2CLS(nn.Module):
    def __init__(self, feat_in, hidden=768, nheads=8, nlayers=6, dropout=0.1):
        super().__init__()
        # from transformer import TransformerEncoderLayer
        self.feat_in = feat_in
        self.cls_token_pre = nn.Parameter(torch.randn(1, 1, feat_in))  #用这个token来聚合pre特征
        self.cls_token_fut = nn.Parameter(torch.randn(1, 1, feat_in))  #用这个token来聚合pre特征
        self.pos_encoder = PositionalEncoding(feat_in, dropout) #根据feat_in生成位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_in, nhead=nheads, dropout=dropout)
        # encoder_layer = TransformerEncoderLayer(d_model=feat_in, nhead=nheads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers, norm=nn.LayerNorm(feat_in))
        self.mlp_pre = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_in, hidden))
        self.mlp_fut = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_in, hidden))

    def forward(self, feats, key_padding_mask=None):
        # input: bs, seq, feat_in; output: bs, hidden
        b, n, _ = feats.shape
        cls_tokens_pre = repeat(self.cls_token_pre, '() n d -> b n d', b=b)
        cls_tokens_fut = repeat(self.cls_token_fut, '() n d -> b n d', b=b)
        # feats = torch.cat((cls_tokens_pre, cls_tokens_fut, feats), dim=1).permute(1, 0, 2)
        feats = torch.cat((cls_tokens_pre, feats, cls_tokens_fut), dim=1).permute(1, 0, 2)
        feats = self.pos_encoder(feats)
        if key_padding_mask is not None:
            cls_mask = key_padding_mask.new_zeros(b, 1)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask, cls_mask], dim=1)
            feats = self.transformer_encoder(feats, src_key_padding_mask=key_padding_mask)
        else:
            feats = self.transformer_encoder(feats)
        # feats = self.transformer_encoder(feats)
        # if output_all:
        #     return self.mlp(feats.permute(1, 0, 2))
        feats_pre = self.mlp_pre(feats[0])
        feats_fut = self.mlp_fut(feats[-1])
        return feats_pre, feats_fut


class FusionLinear2(nn.Module):
    def __init__(self, modality_in, hidden=768, feat_out=1024, num_class=2513, nheads=8, nlayers=6, dropout=0.5):
        super(FusionLinear2, self).__init__()
        self.num_class = num_class
        self.feat_out = feat_out
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden*4, self.num_class)
        )

    def forward(self, feats):
        action = self.fc(feats)
        return action