"""Main training/test program for RULSTM"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random

from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
from models import RULSTM, SingleTransformer, FusionTransformer, FusionLinear, SingleTransformerPretrain, \
    SingleTransformerClassifier
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from utils import topk_accuracy, ValueMeter, topk_accuracy_multiple_timesteps, get_marginal_indexes, marginalize, softmax,  topk_recall_multiple_timesteps, tta, predictions_to_json, MeanTopKRecallMeter
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import logging
import math
import shutil
import itertools

import pdb


pd.options.display.float_format = '{:05.2f}'.format

def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = ArgumentParser(description="Training program for RULSTM")
parser.add_argument('mode', type=str, choices=['train', 'validate', 'test', 'test', 'validate_json'], default='train',
                    help="Whether to perform training, validation or test.\
                            If test is selected, --json_directory must be used to provide\
                            a directory in which to save the generated jsons.")
parser.add_argument('path_to_data', type=str,
                    help="Path to the data folder, \
                            containing all LMDB datasets")
parser.add_argument('path_to_models', type=str,
                    help="Path to the directory where to save all models")
parser.add_argument('--path_to_pretrain', type=none_or_str, default='/disk/wkj/output/transformer_pretrain/2layer_8head/',
                    help='path to checkpoint pretrained models')
parser.add_argument('--alpha', type=float, default=0.25,
                    help="Distance between time-steps in seconds")
parser.add_argument('--S_enc', type=int, default=6,
                    help="Number of encoding steps. \
                            If early recognition is performed, \
                            this value is discarded.") #本任务没有用！只是datasets封装完整不便修改。
parser.add_argument('--S_ant', type=int, default=8,
                    help="Number of anticipation steps. \
                            If early recognition is performed, \
                            this is the number of frames sampled for each action.")
parser.add_argument('--nheads', type=int, default=8)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--task', type=str, default='early_recognition')
parser.add_argument('--img_tmpl', type=str,
                    default='frame_{:010d}.jpg', help='Template to use to load the representation of a given frame')
parser.add_argument('--sequence_completion', action='store_true',
                    help='A flag to selec sequence completion pretraining rather than standard training.\
                            If not selected, a valid checkpoint for sequence completion pretraining\
                            should be available unless --ignore_checkpoints is specified')
parser.add_argument('--num_class', type=int, default=2513,
                    help='Number of classes')
parser.add_argument('--hidden', type=int, default=768,
                    help='Number of hidden units')
parser.add_argument('--feats_in', type=int, nargs='+', default=[1024, 1024, 352, 1024],
                    help='Input sizes when the fusion modality is selected.')
parser.add_argument('--modality_in', type=int, default=4)
parser.add_argument('--feat_out', type=int, default=2048)
parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
parser.add_argument('--fusion_dropout', type=float, default=0.3, help="Fusion layer Dropout rate")
parser.add_argument('--weight_decay', type=float, default=0.1, help="weight_decay")
parser.add_argument('--batch_size', type=int, default=256, help="Batch Size")
parser.add_argument('--num_workers', type=int, default=4,
                    help="Number of parallel thread to fetch the data")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=1e-6, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
parser.add_argument('--display_every', type=int, default=50,
                    help="Display every n iterations")
parser.add_argument('--epochs', type=int, default=30, help="Training epochs")
parser.add_argument('--warmup', type=float, default=2, help="Training epochs")
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--visdom', action='store_true',
                    help="Whether to log using visdom")
parser.add_argument('--ignore_checkpoints', action='store_true',
                    help='If specified, avoid loading existing models (no pre-training)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--weight_ce_pp', type=float, default=1.)
parser.add_argument('--weight_ce_pf', type=float, default=1.)
parser.add_argument('--weight_ce_fp', type=float, default=1.)
parser.add_argument('--weight_ce_ff', type=float, default=1.)
# parser.add_argument('--triplet_margin', type=float)
# parser.add_argument('--weight_con', type=float, default=1.)
parser.add_argument('--weight_mse_pre', type=float, default=5.)
parser.add_argument('--weight_mse_fut', type=float, default=5.)
parser.add_argument('--weight_mse_by_time', type=float, default=0.0)
parser.add_argument('--weight_late_fusion', type=float, default=1.0)
# parser.add_argument('--kfold_mode', type=str, choices=['skf', 'kf'], default='kf')
# parser.add_argument('--label_smoothing', type=float, default=0)
# parser.add_argument('--weight_con_by_time', type=float, default=1./6.)
# parser.add_argument('--epochs_full', type=int, default=0)
# parser.add_argument('--sample_rate', type=float, default=1.0)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
exp_name = f"Transformer-{args.task}_{args.alpha}_{args.S_ant}"
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = None

if args.sequence_completion:
    exp_name += '_sequence_completion'

if args.visdom:
    # if visdom is required
    # load visdom loggers from torchent
    from torchnet.logger import VisdomPlotLogger, VisdomSaver
    # define loss and accuracy logger
    visdom_loss_logger = VisdomPlotLogger('line', env=exp_name, opts={
                                          'title': 'Loss', 'legend': ['training', 'validation']})
    visdom_accuracy_logger = VisdomPlotLogger('line', env=exp_name, opts={
                                              'title': 'Top5 Acc@1s', 'legend': ['training', 'validation']})
    # define a visdom saver to save the plots
    visdom_saver = VisdomSaver(envs=[exp_name])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def get_loader(mode):
    path_to_lmdb = [join(args.path_to_data, m) for m in ['rgb', 'flow', 'obj', 'audio']]
    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
        'time_step': args.alpha,
        'img_tmpl': args.img_tmpl,
        'action_samples': args.S_ant, 
        'past_features': False,
        'sequence_length': args.S_enc + args.S_ant,
        'label_type': 'action',
        'challenge': 'test' in mode
    }

    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training')


def get_model(pretrain=True):
    if args.path_to_pretrain is not None and pretrain:
        # import pdb; pdb.set_trace()
        rgb_model = SingleTransformerPretrain(args.feats_in[0], args.hidden, args.nheads, args.nlayers, args.dropout)
        flow_model = SingleTransformerPretrain(args.feats_in[1], args.hidden, args.nheads, args.nlayers, args.dropout)
        obj_model = SingleTransformerPretrain(args.feats_in[2], args.hidden, args.nheads, args.nlayers, args.dropout)
        audio_model = SingleTransformerPretrain(args.feats_in[3], args.hidden, args.nheads, args.nlayers, args.dropout)
        logger.info(rgb_model.load_state_dict(torch.load(
            os.path.join(args.path_to_pretrain, 'transformer_pretrain_rgb_best.pth.tar'))['state_dict']))
        logger.info(flow_model.load_state_dict(torch.load(
            os.path.join(args.path_to_pretrain, 'transformer_pretrain_flow_best.pth.tar'))['state_dict']))
        logger.info(obj_model.load_state_dict(torch.load(
            os.path.join(args.path_to_pretrain, 'transformer_pretrain_obj_best.pth.tar'))['state_dict']))
        logger.info(audio_model.load_state_dict(torch.load(
            os.path.join(args.path_to_pretrain, 'transformer_pretrain_audio_best.pth.tar'))['state_dict']))
        rgb_model = rgb_model.transformer_encoder
        flow_model = flow_model.transformer_encoder
        obj_model = obj_model.transformer_encoder
        audio_model = audio_model.transformer_encoder
        # rgb_model = rgb_model.single_transformer
        # flow_model = flow_model.single_transformer
        # obj_model = obj_model.single_transformer
        # audio_model = audio_model.single_transformer
        # rgb_model = SingleTransformer(args.feats_in[0], args.hidden, args.nheads, args.nlayers, args.dropout)
        # flow_model = SingleTransformer(args.feats_in[1], args.hidden, args.nheads, args.nlayers, args.dropout)
        # obj_model = SingleTransformer(args.feats_in[2], args.hidden, args.nheads, args.nlayers, args.dropout)
        # audio_model = SingleTransformer(args.feats_in[3], args.hidden, args.nheads, args.nlayers, args.dropout)
        # logger.info(rgb_model.load_state_dict(torch.load(
        #     os.path.join(args.path_to_pretrain, 'transformer_pretrain_cl_rgb_best.pth.tar'))['state_dict']))
        # logger.info(flow_model.load_state_dict(torch.load(
        #     os.path.join(args.path_to_pretrain, 'transformer_pretrain_cl_flow_best.pth.tar'))['state_dict']))
        # logger.info(obj_model.load_state_dict(torch.load(
        #     os.path.join(args.path_to_pretrain, 'transformer_pretrain_cl_obj_best.pth.tar'))['state_dict']))
        # logger.info(audio_model.load_state_dict(torch.load(
        #     os.path.join(args.path_to_pretrain, 'transformer_pretrain_cl_audio_best.pth.tar'))['state_dict']))
    else:
        rgb_model = SingleTransformer(args.feats_in[0], args.hidden, args.nheads, args.nlayers, args.dropout)
        flow_model = SingleTransformer(args.feats_in[1], args.hidden, args.nheads, args.nlayers, args.dropout)
        obj_model = SingleTransformer(args.feats_in[2], args.hidden, args.nheads, args.nlayers, args.dropout)
        audio_model = SingleTransformer(args.feats_in[3], args.hidden, args.nheads, args.nlayers, args.dropout)

    fusion_model = FusionLinear(args.modality_in, args.hidden, args.feat_out, args.num_class, args.nheads, args.nlayers, args.fusion_dropout)
    classifiers = [torch.nn.Sequential(
        torch.nn.Dropout(args.fusion_dropout),
        torch.nn.Linear(args.hidden, args.num_class)
    ) for _ in range(4)]

    return [rgb_model, flow_model, obj_model, audio_model], fusion_model, classifiers


def load_checkpoint(model, best=False, model_name='pre_fusion'):
    if best:
        chk = torch.load(join(args.path_to_models, exp_name + '_' + model_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(args.path_to_models, exp_name + '_' + model_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(model, epoch, perf, best_perf, is_best=False, model_name='pre_fusion'):
    # pass
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.path_to_models, exp_name + '_' + model_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.path_to_models, exp_name + '_' + model_name + '_best.pth.tar'))

    if args.visdom:
        # save visdom logs for persitency
        visdom_saver.save()


def get_avg(accuracy_meter):
    acc = 0
    for k in range(8):
        acc_k = accuracy_meter[k].value()
        acc += acc_k
    acc = acc/8
    return acc


def log(mode, epoch, loss_meter, accuracy_pre_meter, accuracy_fut_meter, mse_pre_meter=None, mse_fut_meter=None,
        best_perf=None, green=False):
    if green:
        raise NotImplementedError
        # logger.info('\033[92m', end="")
    msg = f"[{mode}] Epoch: {epoch:0.2f}. Loss: {loss_meter.value():.2f}. "
    acc_pre = 0
    acc_fut = 0
    msg_pre = ''
    msg_fut = ''
    for k in range(8):
        acc_k_pre = accuracy_pre_meter[k].value()
        acc_k_fut = accuracy_fut_meter[k].value()
        msg_pre += f'{12.5*(k+1):.1f}%: {acc_k_pre:.2f}%, '
        msg_fut += f'{12.5*(8-k):.1f}%: {acc_k_fut:.2f}%, '
        acc_pre += acc_k_pre
        acc_fut += acc_k_fut
    acc_pre = acc_pre/8
    acc_fut = acc_fut/8

    if best_perf:
        logger.info(f"{msg}[PRE:{msg_pre}][FUT:{msg_fut}]Avg: {acc_pre: .2f}%/{acc_fut: .2f}%[best: {best_perf:0.2f}%]")
    else:
        logger.info(f"{msg}[PRE:{msg_pre}][FUT:{msg_fut}]Avg: {acc_pre: .2f}%/{acc_fut: .2f}%")

    if mse_pre_meter is not None and mse_fut_meter is not None:
        mse_pre = 0
        mse_fut = 0
        msg_pre = ''
        msg_fut = ''
        for k in range(7):
            mse_k_pre = mse_pre_meter[k].value()
            mse_k_fut = mse_fut_meter[k].value()
            msg_pre += f'{12.5*(k+1):.1f}%: {mse_k_pre:.2f}, '
            msg_fut += f'{12.5*(k+1):.1f}%: {mse_k_fut:.2f}, '
            mse_pre += mse_k_pre
            mse_fut += mse_k_fut
        mse_pre = mse_pre/7
        mse_fut = mse_fut/7
        logger.info(f"[{mode}] Epoch: {epoch:0.2f}. MSE(pre): {mse_pre: .2f}. {msg_pre}; MSE(fut): {mse_fut: .2f}. {msg_fut}.")
    # print('\033[0m')
    #
    # if args.visdom:
    #     visdom_loss_logger.log(epoch, loss_meter.value(), name=mode)
    #     visdom_accuracy_logger.log(epoch, accuracy_meter.value(), name=mode)

def get_scores(pre_models, pre_fusion_model, loader, k):
    pre_fusion_model.eval()
    for i in range(args.modality_in):
        pre_models[i].eval()
    predictions = []
    labels = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['action_features']
            x[0] = (x[0] - 0.786) / 1.532
            x[1] = (x[1] - 1.033) / 2.666
            x[2] = (x[2] - 0.006) / 0.065
            x[3] = (x[3] - 1.086) / 2.642
            x = [xx.to(device) for xx in x]
            # x = [lstm_models[i](x[i]) for i in range(args.modality_in)]
            y = batch['label'].numpy()

            x1 = [xx[:, :k, :] for xx in x]
            pre_features = [pre_models[j](x1[j]) for j in range(4)]
            pre_features = torch.stack(pre_features, 1)
            apre, hpre, hfut_ = pre_fusion_model(pre_features, 'pre')
            apre = apre.cpu().numpy()
            predictions.append(apre)
            labels.append(y)

    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc = topk_accuracy(action_scores, labels, (1,))[0]*100
    return(acc)


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def trainval(
        pre_models, pre_fusion_model, pre_classifiers, fut_models, fut_fusion_model, fut_classifiers, loaders,
        optimizer_pre, optimizer_fut, epochs, scheduler_pre=None, scheduler_fut=None):
    """Training/Validation code"""
    best_perf = 0
    for epoch in range(epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        mse_pre_meter = {'training': [ValueMeter() for _ in range(7)], 'validation': [ValueMeter() for _ in range(7)]}
        mse_fut_meter = {'training': [ValueMeter() for _ in range(7)], 'validation': [ValueMeter() for _ in range(7)]}
        accuracy_pre_meter = {'training': [ValueMeter() for _ in range(8)], 'validation': [ValueMeter() for _ in range(8)]}
        accuracy_pre_meter_ens = {'training': [ValueMeter() for _ in range(8)], 'validation': [ValueMeter() for _ in range(8)]}
        accuracy_fut_meter = {'training': [ValueMeter() for _ in range(8)], 'validation': [ValueMeter() for _ in range(8)]}
        accuracy_fut_meter_ens = {'training': [ValueMeter() for _ in range(8)], 'validation': [ValueMeter() for _ in range(8)]}
        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    for i in range(args.modality_in):
                        pre_models[i].train()
                        pre_classifiers[i].train()
                        fut_models[i].train()
                        fut_classifiers[i].train()
                    pre_fusion_model.train()
                    fut_fusion_model.train()
                    # import pdb; pdb.set_trace()
                    for i, batch in enumerate(loaders[mode]):
                        x = batch['action_features']
                        x[0] = (x[0] - 0.786) / 1.532
                        x[1] = (x[1] - 1.033) / 2.666
                        x[2] = (x[2] - 0.006) / 0.065
                        x[3] = (x[3] - 1.086) / 2.642
                        x = [xx.to(device) for xx in x]

                        # 先训练一边pre-model
                        losses_pre = []
                        for k in range(1, 9):
                            x1 = [xx[:, :k, :] for xx in x]
                            x2 = [xx[:, k:, :] for xx in x]
                            y = batch['label'].to(device)

                            bs = y.shape[0]  # batch size

                            # pre_model计算
                            pre_features = [pre_models[j](x1[j]) for j in range(4)]
                            apre_rgb = pre_classifiers[0](pre_features[0])
                            apre_flow = pre_classifiers[1](pre_features[1])
                            apre_obj = pre_classifiers[2](pre_features[2])
                            apre_audio = pre_classifiers[3](pre_features[3])
                            loss_early_p = 0.25 * (
                                     F.cross_entropy(apre_rgb, y) +
                                     F.cross_entropy(apre_flow, y) +
                                     F.cross_entropy(apre_obj, y) +
                                     F.cross_entropy(apre_audio, y)
                            )

                            pre_features = torch.stack(pre_features, 1)
                            # fut_model计算
                            fut_features = [fut_models[j](x2[j]) for j in range(4)]
                            fut_features = torch.stack(fut_features, 1)

                            if k == 8:
                                # 只训练pre-model
                                app, hpp, hpf = pre_fusion_model(pre_features, 'pre')
                                apf = fut_fusion_model.fc(hpf)

                                loss_ce_pp = F.cross_entropy(app, y)
                                loss_ce_pf = F.cross_entropy(apf, y)

                                loss_p = args.weight_ce_pp * loss_ce_pp + args.weight_ce_pf * loss_ce_pf +\
                                         args.weight_late_fusion * loss_early_p
                                losses_pre.append(loss_p)
                            else:
                                # 第一次正向传播, 训练pre-model
                                app, hpp, hpf = pre_fusion_model(pre_features, 'pre')
                                aff, hfp, hff = fut_fusion_model(fut_features, 'fut')

                                # 用fut-model的最后的fc层对pre-model输出的pf分类
                                apf = fut_fusion_model.fc(hpf)

                                loss_ce_pp = F.cross_entropy(app, y)
                                loss_ce_pf = F.cross_entropy(apf, y)

                                loss_mse_pre = F.mse_loss(hpp, hfp.detach())
                                loss_mse_fut = F.mse_loss(hpf, hff.detach())

                                coef = 1 + args.weight_mse_by_time * (abs(4 - k) - 12 / 7)

                                loss_p = args.weight_ce_pp * loss_ce_pp + args.weight_ce_pf * loss_ce_pf + \
                                         coef * (args.weight_mse_pre * loss_mse_pre + args.weight_mse_fut * loss_mse_fut) + \
                                         args.weight_late_fusion * loss_early_p
                                losses_pre.append(loss_p)

                                mse_pre_meter[mode][k-1].add(loss_mse_pre.item(), bs)
                                mse_fut_meter[mode][k-1].add(loss_mse_fut.item(), bs)

                            acc_pre = topk_accuracy(
                                app.detach().cpu().numpy(), y.detach().cpu().numpy(), (1,))[0]*100
                            accuracy_pre_meter[mode][k-1].add(acc_pre, bs)

                            app_ens = 0.5 * app + 0.5 * 0.25 * (apre_rgb + apre_flow + apre_obj + apre_audio)
                            acc_pre_ens = topk_accuracy(
                                app_ens.detach().cpu().numpy(), y.detach().cpu().numpy(), (1,))[0]*100
                            accuracy_pre_meter_ens[mode][k-1].add(acc_pre_ens, bs)
                        # pdb.set_trace()
                        loss_pre = sum(losses_pre)/8

                        optimizer_pre.zero_grad()
                        loss_pre.backward()
                        optimizer_pre.step()

                        loss_meter[mode].add(loss_pre.item(), bs)

                        # 训练一边fut-model
                        losses_fut = []
                        for k in range(0, 8):
                            x1 = [xx[:, :k, :] for xx in x]
                            x2 = [xx[:, k:, :] for xx in x]
                            y = batch['label'].to(device)

                            bs = y.shape[0]  # batch size

                            # pre_model计算
                            pre_features = [pre_models[j](x1[j]) for j in range(4)]
                            pre_features = torch.stack(pre_features, 1)
                            # fut_model计算
                            fut_features = [fut_models[j](x2[j]) for j in range(4)]
                            afut_rgb = fut_classifiers[0](fut_features[0])
                            afut_flow = fut_classifiers[1](fut_features[1])
                            afut_obj = fut_classifiers[2](fut_features[2])
                            afut_audio = fut_classifiers[3](fut_features[3])
                            loss_early_f = 0.25 * (
                                     F.cross_entropy(afut_rgb, y) +
                                     F.cross_entropy(afut_flow, y) +
                                     F.cross_entropy(afut_obj, y) +
                                     F.cross_entropy(afut_audio, y)
                            )
                            fut_features = torch.stack(fut_features, 1)

                            if k == 0:
                                # 只训练fut-model
                                aff, hfp, hff = fut_fusion_model(fut_features, 'fut')
                                afp = pre_fusion_model.fc(hfp)

                                loss_ce_fp = F.cross_entropy(afp, y)
                                loss_ce_ff = F.cross_entropy(aff, y)

                                loss_f = args.weight_ce_ff * loss_ce_ff + args.weight_ce_fp * loss_ce_fp + \
                                         args.weight_late_fusion * loss_early_f
                                losses_fut.append(loss_f)
                            else:
                                # 第二次正向传播, 训练fut-model
                                app, hpp, hpf = pre_fusion_model(pre_features, 'pre')
                                aff, hfp, hff = fut_fusion_model(fut_features, 'fut')

                                # 用pre-model的最后的fc层对fut-model输出的fp分类
                                afp = pre_fusion_model.fc(hfp)

                                loss_ce_fp = F.cross_entropy(afp, y)
                                loss_ce_ff = F.cross_entropy(aff, y)

                                loss_mse_pre = F.mse_loss(hpp.detach(), hfp)
                                loss_mse_fut = F.mse_loss(hpf.detach(), hff)

                                coef = 1 + args.weight_mse_by_time * (abs(4 - k) - 12 / 7)

                                loss_f = args.weight_ce_ff * loss_ce_ff + args.weight_ce_fp * loss_ce_fp + \
                                         coef * (args.weight_mse_pre * loss_mse_pre + args.weight_mse_fut * loss_mse_fut) + \
                                         args.weight_late_fusion * loss_early_f
                                losses_fut.append(loss_f)

                                mse_pre_meter[mode][k-1].add(loss_mse_pre.item(), bs)
                                mse_fut_meter[mode][k-1].add(loss_mse_fut.item(), bs)

                            acc_fut = topk_accuracy(
                                aff.detach().cpu().numpy(), y.detach().cpu().numpy(), (1,))[0] * 100
                            accuracy_fut_meter[mode][k].add(acc_fut, bs)
                            aff_ens = 0.5 * aff + 0.5 * 0.25 * (afut_rgb + afut_flow + afut_obj + afut_audio)
                            acc_fut_ens = topk_accuracy(
                                aff_ens.detach().cpu().numpy(), y.detach().cpu().numpy(), (1,))[0]*100
                            accuracy_fut_meter_ens[mode][k-1].add(acc_fut_ens, bs)

                        loss_fut = sum(losses_fut)/8

                        optimizer_fut.zero_grad()
                        loss_fut.backward()
                        optimizer_fut.step()
                        loss_meter[mode].add(loss_fut.item(), bs)

                        e = epoch + i/len(loaders[mode])

                        if i != 0 and i % args.display_every == 0:
                            log(mode, e, loss_meter[mode], accuracy_pre_meter_ens[mode], accuracy_fut_meter_ens[mode])
                            log(mode, e, loss_meter[mode], accuracy_pre_meter[mode], accuracy_fut_meter[mode],
                                mse_pre_meter[mode], mse_fut_meter[mode])
                else:
                    sum_accuracy_pre = 0
                    sum_accuracy_pre_ens = 0
                    sum_accuracy_fut = 0
                    sum_accuracy_fut_ens = 0
                    for i in range(args.modality_in):
                        pre_models[i].eval()
                        pre_classifiers[i].eval()
                        fut_models[i].eval()
                        fut_classifiers[i].eval()
                    pre_fusion_model.eval()
                    fut_fusion_model.eval()
                    for k in range(0, 8):
                        predictions_pre = []
                        predictions_pre_ens = []
                        predictions_fut = []
                        predictions_fut_ens = []
                        labels = []
                        for i, batch in enumerate(loaders[mode]):
                            x = batch['action_features']
                            x[0] = (x[0] - 0.786) / 1.532
                            x[1] = (x[1] - 1.033) / 2.666
                            x[2] = (x[2] - 0.006) / 0.065
                            x[3] = (x[3] - 1.086) / 2.642

                            x = [xx.to(device) for xx in x]
                            # x = [lstm_models[i](x[i]) for i in range(args.modality_in)]
                            x1 = [xx[:, :k+1, :] for xx in x]
                            x2 = [xx[:, k:, :] for xx in x]
                            y = batch['label'].numpy()

                            pre_features = [pre_models[j](x1[j]) for j in range(4)]
                            apre_rgb = pre_classifiers[0](pre_features[0])
                            apre_flow = pre_classifiers[1](pre_features[1])
                            apre_obj = pre_classifiers[2](pre_features[2])
                            apre_audio = pre_classifiers[3](pre_features[3])

                            pre_features = torch.stack(pre_features, 1)

                            apre, _, hpf = pre_fusion_model(pre_features, 'pre')
                            apf = fut_fusion_model.fc(hpf)
                            apre = 0.5 * (apre + apf)
                            apre_ens = 0.5 * apre + 0.5 * 0.25 * (apre_rgb + apre_flow + apre_obj + apre_audio)
                            apre = apre.cpu().numpy()
                            apre_ens = apre_ens.cpu().numpy()
                            predictions_pre.append(apre)
                            predictions_pre_ens.append(apre_ens)

                            fut_features = [fut_models[j](x2[j]) for j in range(4)]
                            afut_rgb = fut_classifiers[0](fut_features[0])
                            afut_flow = fut_classifiers[1](fut_features[1])
                            afut_obj = fut_classifiers[2](fut_features[2])
                            afut_audio = fut_classifiers[3](fut_features[3])

                            fut_features = torch.stack(fut_features, 1)

                            afut, hfp, _ = fut_fusion_model(fut_features, 'fut')
                            afp = pre_fusion_model.fc(hfp)
                            afut = 0.5 * (afut + afp)
                            afut_ens = 0.5 * afut + 0.5 * 0.25 * (afut_rgb + afut_flow + afut_obj + afut_audio)
                            afut = afut.cpu().numpy()
                            afut_ens = afut_ens.cpu().numpy()
                            predictions_fut.append(afut)
                            predictions_fut_ens.append(afut_ens)

                            labels.append(y)

                        action_scores_pre = np.concatenate(predictions_pre)
                        action_scores_pre_ens = np.concatenate(predictions_pre_ens)
                        action_scores_fut = np.concatenate(predictions_fut)
                        action_scores_fut_ens = np.concatenate(predictions_fut_ens)
                        labels = np.concatenate(labels)
                        acc_pre = topk_accuracy(action_scores_pre, labels, (1,))[0]*100
                        acc_pre_ens = topk_accuracy(action_scores_pre_ens, labels, (1,))[0]*100
                        acc_fut = topk_accuracy(action_scores_fut, labels, (1,))[0]*100
                        acc_fut_ens = topk_accuracy(action_scores_fut_ens, labels, (1,))[0]*100

                        sum_accuracy_pre += acc_pre
                        sum_accuracy_pre_ens += acc_pre_ens
                        sum_accuracy_fut += acc_fut
                        sum_accuracy_fut_ens += acc_fut_ens
                        logger.info('\033[92m'
                                      f"Epoch: {epoch + 1:0.2f}. "
                                      f"Accuracy {float(k+1)/8*100:0.1f}%: {acc_pre:0.2f}%. "
                                      f"Accuracy(Ens) {float(k+1)/8*100:0.1f}%: {acc_pre_ens:0.2f}%. "
                                      f"Accuracy {float(8-k)/8*100:0.1f}%: {acc_fut:0.2f}%. "
                                      f"Accuracy(Ens) {float(8-k)/8*100:0.1f}%: {acc_fut_ens:0.2f}%. "
                                      '\033[0m')
                        if k == 7:
                            avg_accuracy_pre = sum_accuracy_pre / 8
                            avg_accuracy_pre_ens = sum_accuracy_pre_ens / 8
                            avg_accuracy_fut = sum_accuracy_fut / 8
                            avg_accuracy_fut_ens = sum_accuracy_fut_ens / 8
                            logger.info('\033[92m'
                                          f"[{mode}] Epoch: {epoch + 1:0.2f}. "
                                          f"Avg_Pre_Accuracy: {avg_accuracy_pre:.2f}%. "
                                          f"Avg_Pre_Accuracy(Ens): {avg_accuracy_pre_ens:.2f}%. "
                                          f"Avg_Fut_Accuracy: {avg_accuracy_fut:.2f}%. "
                                          f"Avg_Fut_Accuracy(Ens): {avg_accuracy_fut_ens:.2f}%. "
                                          f"[best: {max(avg_accuracy_pre_ens, best_perf):0.2f}]%"
                                          '\033[0m')

                            if best_perf < avg_accuracy_pre_ens:
                                best_perf = avg_accuracy_pre_ens
                                is_best = True
                            else:
                                is_best = False

                            for i, modality in enumerate(['rgb', 'flow', 'obj', 'audio']):
                                save_model(pre_models[i], epoch+1, avg_accuracy_pre_ens, best_perf, is_best=is_best, model_name=f'pre_{modality}')
                                save_model(pre_classifiers[i], epoch+1, avg_accuracy_pre_ens, best_perf, is_best=is_best, model_name=f'pre_{modality}_classifier')
                                save_model(fut_models[i], epoch+1, avg_accuracy_pre_ens, best_perf, is_best=is_best, model_name=f'fut_{modality}')
                                save_model(fut_classifiers[i], epoch+1, avg_accuracy_pre_ens, best_perf, is_best=is_best, model_name=f'fut_{modality}_classifier')
                            save_model(pre_fusion_model, epoch+1, avg_accuracy_pre_ens, best_perf, is_best=is_best, model_name='pre_fusion')
                            save_model(fut_fusion_model, epoch+1, avg_accuracy_pre_ens, best_perf, is_best=is_best, model_name='fut_fusion')
                    # if mode == 'training':

        if scheduler_pre is not None:
            scheduler_pre.step()
        if scheduler_fut is not None:
            scheduler_fut.step()

    logger.info('Final Validation......')
    load_checkpoint(pre_fusion_model, True, 'pre_fusion')
    pre_fusion_model.eval()
    load_checkpoint(fut_fusion_model, True, 'fut_fusion')
    fut_fusion_model.eval()
    for j, modal in enumerate(['rgb', 'flow', 'obj', 'audio']):
        load_checkpoint(pre_models[j], True, f'pre_{modal}')
        pre_models[j].eval()
        load_checkpoint(pre_classifiers[j], True, f'pre_{modal}_classifier')
        pre_classifiers[j].eval()
    best_perfs = []
    best_weights = []
    predictions = []
    with torch.set_grad_enabled(False):
        for k in range(0, 8):
            predictions_pre_rgb = []
            predictions_pre_flow = []
            predictions_pre_obj = []
            predictions_pre_audio = []
            predictions_pre_ens = []
            labels = []
            for i, batch in enumerate(loaders['validation']):
                x = batch['action_features']
                x[0] = (x[0] - 0.786) / 1.532
                x[1] = (x[1] - 1.033) / 2.666
                x[2] = (x[2] - 0.006) / 0.065
                x[3] = (x[3] - 1.086) / 2.642

                x = [xx.to(device) for xx in x]
                # x = [lstm_models[i](x[i]) for i in range(args.modality_in)]
                x1 = [xx[:, :k + 1, :] for xx in x]
                y = batch['label'].numpy()

                pre_features = [pre_models[j](x1[j]) for j in range(4)]
                apre_rgb = pre_classifiers[0](pre_features[0])
                apre_flow = pre_classifiers[1](pre_features[1])
                apre_obj = pre_classifiers[2](pre_features[2])
                apre_audio = pre_classifiers[3](pre_features[3])

                pre_features = torch.stack(pre_features, 1)

                apre, _, hpf = pre_fusion_model(pre_features, 'pre')
                apf = fut_fusion_model.fc(hpf)
                apre_ens = 0.5 * (apre + apf)
                predictions_pre_ens.append(apre_ens.cpu().numpy())

                predictions_pre_rgb.append(apre_rgb.cpu().numpy())
                predictions_pre_flow.append(apre_flow.cpu().numpy())
                predictions_pre_obj.append(apre_obj.cpu().numpy())
                predictions_pre_audio.append(apre_audio.cpu().numpy())
                labels.append(y)

            action_scores_pre_ens = np.concatenate(predictions_pre_ens)
            action_scores_pre_rgb = np.concatenate(predictions_pre_rgb)
            action_scores_pre_flow = np.concatenate(predictions_pre_flow)
            action_scores_pre_obj = np.concatenate(predictions_pre_obj)
            action_scores_pre_audio = np.concatenate(predictions_pre_audio)
            labels = np.concatenate(labels)

            predictions.append({
                'ens': action_scores_pre_ens,
                'rgb': action_scores_pre_rgb,
                'flow': action_scores_pre_flow,
                'obj': action_scores_pre_obj,
                'audio': action_scores_pre_audio,
                'labels': labels
            })
    accs = []
    # for wrgb in (0., 0.1, 0.2):
    #     for wflow in (0., 0.1, 0.2):
    #         for wobj in (0., 0.1, 0.2):
    #             for waudio in (0., 0.1, 0.2):
    for wrgb in (0.02, ):
        for wflow in (0.10,):
            for wobj in (0.18,):
                for waudio in (0.22, ):
                    acc = 0
                    wearly = 1 - wrgb - wflow - wobj - waudio
                    for k in range(8):
                        action_scores = wearly * predictions[k]['ens'] + wrgb * predictions[k]['rgb'] +\
                                        wflow * predictions[k]['flow'] + wobj * predictions[k]['obj'] +\
                                        waudio * predictions[k]['audio']
                        labels = predictions[k]['labels']
                        acc_k = (action_scores.argmax(-1) == labels).mean()
                        print(f'{acc_k*100:.2f}')
                        acc += acc_k
                    acc = acc / 8
                    logger.info(f'{wearly:.2f}, {wrgb:.2f}, {wflow:.2f}, {wobj:.2f}, {waudio:.2f}, {acc*100:.2f}')
                    accs.append((acc, wrgb, wflow, wobj, waudio))
    best_perf, wrgb, wflow, wobj, waudio = max(accs)
    wearly = 1 - wrgb - wflow - wobj - waudio
    logger.info(f'\033[92m Final Validation Avg_Pre_Accuracy: {best_perf*100:.2f}%'
                f'({wearly:.2f}, {wrgb:.2f}, {wflow:.2f}, {wobj:.2f}, {waudio:.2f}).\033[0m')


def cosine_annealing_with_warmup(warm_up_iter, max_iter, lr_min, lr_max):
    def fun(cur_iter):
        if cur_iter < warm_up_iter:
            return (cur_iter+1)/warm_up_iter
        else:
            return (lr_min + 0.5 * (lr_max - lr_min)*(1. + math.cos(math.pi * (cur_iter - warm_up_iter)/(max_iter-warm_up_iter))))/lr_max
    return fun


# def get_scheduler(optimizer, warmup, epochs, lr_min, lr):
def get_scheduler(optimizer):
    # return torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 20], 0.5)
    # return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_annealing_with_warmup(
    #     warmup, epochs, lr_min, lr))


def main():
    set_seed(args.seed)
    # 新建目录
    if not os.path.exists(args.path_to_models):
        os.makedirs(args.path_to_models)
    global logger
    logger = logging.getLogger(f'main{random.randint(0, 1e5)}')
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    # logger.removeHandler(logger.handlers)
    logger.addHandler(logging.FileHandler(os.path.join(args.path_to_models, 'log.log')))
    # logging.basicConfig(filename=os.path.join(args.path_to_models, 'log.log'), level=logging.DEBUG)
    # lstm_models, pre_models, pre_fusion_model = get_model()
    pre_models, pre_fusion_model, pre_classifiers = get_model()
    # _, fut_models, fut_fusion_model = get_model()
    fut_models, fut_fusion_model, fut_classifiers = get_model()

    # lstm_models = [m.to(device) for m in lstm_models]
    pre_models = [m.to(device) for m in pre_models]
    fut_models = [m.to(device) for m in fut_models]
    pre_classifiers = [m.to(device) for m in pre_classifiers]
    fut_classifiers = [m.to(device) for m in fut_classifiers]
    pre_fusion_model.to(device)
    fut_fusion_model.to(device)
    
    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}
        optimizer_pre = torch.optim.AdamW(itertools.chain(pre_models[0].parameters(),
                                                          pre_models[1].parameters(),
                                                          pre_models[2].parameters(),
                                                          pre_models[3].parameters(),
                                                          pre_classifiers[0].parameters(),
                                                          pre_classifiers[1].parameters(),
                                                          pre_classifiers[2].parameters(),
                                                          pre_classifiers[3].parameters(),
                                                          pre_fusion_model.parameters()
                                                          ), lr=args.lr, weight_decay=args.weight_decay,
                                          betas=(0.5, 0.999))

        optimizer_fut = torch.optim.AdamW(itertools.chain(fut_models[0].parameters(),
                                                          fut_models[1].parameters(),
                                                          fut_models[2].parameters(),
                                                          fut_models[3].parameters(),
                                                          fut_classifiers[0].parameters(),
                                                          fut_classifiers[1].parameters(),
                                                          fut_classifiers[2].parameters(),
                                                          fut_classifiers[3].parameters(),
                                                          fut_fusion_model.parameters()
                                                          ), lr=args.lr, weight_decay=args.weight_decay,
                                          betas=(0.5, 0.999))

        batch_per_epoch = len(loaders['training'].dataset) // args.batch_size
        if args.lr_scheduler:
            scheduler_pre = get_scheduler(optimizer_pre)
            scheduler_fut = get_scheduler(optimizer_fut)
            trainval(pre_models, pre_fusion_model, pre_classifiers, fut_models, fut_fusion_model, fut_classifiers,
                     loaders, optimizer_pre, optimizer_fut, args.epochs, scheduler_pre, scheduler_fut)
        else:
            trainval(pre_models, pre_fusion_model, pre_classifiers, fut_models, fut_fusion_model, fut_classifiers,
                     loaders, optimizer_pre, optimizer_fut, args.epochs)


    elif args.mode == 'validate':
        for i, modality in enumerate(['rgb', 'flow', 'obj', 'audio']):
            model_name = 'pre_' + modality
            # model_name = 'fut_' + modality
            epoch, perf, _ = load_checkpoint(pre_models[i], best=True, model_name=model_name)
            
        epoch, perf, _ = load_checkpoint(pre_fusion_model, best=True, model_name='pre_fusion')
        # epoch, perf, _ = load_checkpoint(pre_fusion_model, best=True, model_name='fut_fusion')
        logger.info(f"Loaded checkpoint for model {type(pre_fusion_model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")
        loaders = get_loader('validation')
        accs = []
        for k in range(1, 9):
            acc = get_scores(pre_models, pre_fusion_model, loaders, k)
            accs.append(acc)
        for i in range(8):
            logger.info(f"Accuracy of observation ratio {float(i+1)/8*100:0.2f}%: {accs[i]:0.2f}%.")
        logger.info(f"Average Accuracy : {sum(accs)/8:0.2f}%.")


if __name__ == '__main__':
    # args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}_latefusion{args.weight_late_fusion}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    # args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}_latefusion1/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep30_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    # args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_15_20_0.5/alt_ens_weightbytime0.0_latefusion1.0/lr0.0001_bs256_dropout0.3_fusiondropout0.3_wd0.1_ep30_mse5.0_5.0_cepf1.0_cefp1.0_featout_2048_lrschTrue/'
    main()

    # for lr in (1e-4, ):
    #     args.lr = lr
    #     for weight_late_fusion in (1, ):
    #         args.weight_late_fusion = weight_late_fusion
    #         args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}_latefusion{args.weight_late_fusion}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    #         # args.path_to_models = f'{args.path_to_pretrain}/finetune_step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    #         main()
    # args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/alt_ens/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    # args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/alt_ens_ce100/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    # for w in range(1, 7):
    #     ww = float(w)/12
    #     args.weight_mse_by_time = ww
    #     args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    #     main()
    # for fusion_dropout in (0.3,):
    #     args.fusion_dropout = fusion_dropout
    #     args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
        # args.path_to_models = f'{args.path_to_pretrain}/finetune_step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
        # main()
    # for pre_lr in (1e-4, 1e-5):
    #     for pre_dropout in (0, 0.1, 0.3):
    #         args.path_to_pretrain = f'/home/share/wangkejie/Video-Transformer/models/transformer_pretrain_new_mse/' \
    #                                 f'2layer_8head/hidden1536_featout2048/' \
    #                                 f'batchsize128_lr{pre_lr}_wd0_dropout{pre_dropout}_epoch50'
    #         # args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    #         args.path_to_models = f'{args.path_to_pretrain}/finetune_step_15_20_0.5/' \
    #                               f'alt_ens_weightbytime{args.weight_mse_by_time}/' \
    #                               f'lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}' \
    #                               f'_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}' \
    #                               f'_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}' \
    #                               f'_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
            # args.path_to_models = f'{args.path_to_pretrain}/finetune_step_15_20_0.5/alt_ens_weightbytime{args.weight_mse_by_time}/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
            # main()
    # for weight_ce_100 in (2, 4, 8):
    #     args.weight_ce_100 = weight_ce_100
    #     args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/alt_ens_ce100/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_wce100_{args.weight_ce_100}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    #     main()
    # for fusion_dropout in (0, 0.1, 0.3, 0.5):
    #     args.fusion_dropout = fusion_dropout
    #     args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/alt_ens_complexfc/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_fusiondropout{args.fusion_dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    #     main()
    # for featout in (2048, 4096):
        # args.feat_out = featout
        # args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/alt_ens/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
        # main()

    # for w_pf_fp in (1, ):
    #     for w_mse in (0.01, 0.1, 10):
    #         args.weight_mse_pre = args.weight_mse_fut = w_mse
    #         args.weight_ce_pf = args.weight_ce_fp = w_pf_fp
    #         args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/alt_ens_weightbytime_norm/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_cepf{args.weight_ce_pf}_cefp{args.weight_ce_fp}_featout_{args.feat_out}_lrsch{args.lr_scheduler}'
    #         main()
    # main()
    # for lr in (2e-5, 1e-5):
    #     args.lr = lr
    # for dropout in (0.3, ):
    #     args.dropout = dropout
    #     args.lr = 3e-5
    #     args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/alt_ens/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_mse_pre}_{args.weight_mse_fut}_lrsch{args.lr_scheduler}'
    #     main()
    # for wd in (0.01, 0.1, 1., 10., 100.):
    #     for bs in (64, 128):
    # for wd in (0.1, ):
    #     for bs in (64, ):
    #         args.batch_size = bs
    #         args.weight_decay = wd
    #         args.path_to_models = f'../output/preonly/2layer_8head/new_nopad/step_10_15_0.1/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}'
    #         main()
    # for weight_con in (0., ):
    #     args.weight_con = weight_con
    #     args.path_to_models = f'../output/prefut/2layer_8head/new_nopad/step_12_16_0.1/new_validate_lossdiv8/lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}_wd{args.weight_decay}_ep{args.epochs}_mse{args.weight_con}'
    #     main()
#CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python main_simple.py train /disk/gao1/Transformer/data/ek55/ step10_15 --nheads 8 --nlayers 2 --epochs 20 --lr 0.00003 --batch_size 64 --dropout 0.5 --path_to_pretrain ../output/transformer_pretrain/2layer_8head/ --lr_scheduler --weight_decay 0.1
    # for dropout in (0.5, 0.7):
    #     args.dropout = dropout
    #     args.path_to_models = f'../output/preonly/2layer_8head/new_nopad/lr{args.lr}_bs{args.batch_size}_dropout{dropout}'
    # for lr in (0.00003, ):
    #     for bs in (64, 128, 256):
    #         for dropout in (0.1, 0.3, 0.5, 0.7):
    #             args.lr = lr
    #             args.bs = bs
    #             args.dropout = dropout
    #             args.path_to_models = f'../output/preonly/2layer_8head/lr{lr}_bs{bs}_dropout{dropout}'
    #             main()
    # for bs in (32, 64, 128):
    #     for lr in (0.0001,):
    #         args.lr = lr
    #         args.bs = bs
    #         args.path_to_models = f'../output/preonly/2layer_8head/lr{lr}_bs{bs}_full{args.epochs_full}_dropout{args.dropout}'
    #         main()
    # for lr in (0.001, 0.0003, 0.00003, 0.00001):
    #     for wd in (0.1, 1, 10):
    #         args.lr = args.lr_fut = lr
    #         args.weight_decay = wd
    #         args.path_to_models = f'../output/transformer_pretrain/2layer_8head/loss_weight_by_time/mse_linear_dec_inc/droptoken_sample{args.sample_rate}_lr{lr}_wd{wd}'
    #         main()

    # for lr_fut in (0.001, 0.0003, 0.00003, 0.00001):
    #     args.lr_fut = lr_fut
    #     args.path_to_models = f'../output/transformer_pretrain/2layer_8head/loss_weight_by_time/mse_linear_dec_inc/lr_fut{lr_fut}'
    #     main()
    # for w in range(1, 7):
    #     weight = float(w) / 9
    #     args.weight_con_by_time = weight
    #     args.path_to_models = f'../output/transformer_pretrain/2layer_8head/loss_weight_by_time/mse_linear_inc_dec/weight{weight}'
    #     main()
    # for w in range(1, 7):
    #     weight = float(w) / 12
    #     args.weight_con_by_time = weight
    #     args.path_to_models = f'../output/transformer_pretrain/2layer_8head/loss_weight_by_time/mse_linear_dec_inc/weight{weight}'
    #     main()
    # for w in range(1, 8):
    #     weight = float(w) / 32
    #     args.weight_con_by_time = weight
    #     args.path_to_models = f'../output/transformer_pretrain/2layer_8head/loss_weight_by_time/mse_linear_dec_inc_quadratic/weight{weight}'
    #     main()
    # for w in (1, 2, 4, 6, 8, 10):
    #     args.weight_pre = args.weight_fut = w
    #     args.weight_con = 2 * w
    #     args.path_to_models = f'../output/transformer_pretrain/2layer_8head/loss_weight_by_time/decrease/inverse_pre{w}fut{w}con{w*2}'
    #     main()
    # for smooth in (0.02, 0.04, 0.08):
    #     args.label_smoothing = smooth
    #     args.path_to_models = f'../output/transformer_pretrain/2layer_8head/loss_reweight/smooth{smooth}'
    #     main()
