"""Main training/test program for RULSTM"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random

from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
from models import SingleTransformer, FusionLinear
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
parser.add_argument('path_to_models_stage2', type=str)
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
parser.add_argument('--num_class', type=int, default=106,
                    help='Number of classes')
parser.add_argument('--hidden', type=int, default=768,
                    help='Number of hidden units')
parser.add_argument('--feats_in', type=int, nargs='+', default=[1024, 1024, 352],
                    help='Input sizes when the fusion modality is selected.')
parser.add_argument('--modality_in', type=int, default=3)
parser.add_argument('--feat_out', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
parser.add_argument('--fusion_dropout', type=float, default=0.9, help="Fusion layer Dropout rate")
parser.add_argument('--weight_decay', type=float, default=0.1, help="weight_decay")
parser.add_argument('--batch_size', type=int, default=1024, help="Batch Size")
parser.add_argument('--num_workers', type=int, default=4,
                    help="Number of parallel thread to fetch the data")
parser.add_argument('--lr', type=float, default=0.0007, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=1e-6, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9 , help="Momentum")
parser.add_argument('--display_every', type=int, default=10,
                    help="Display every n iterations")
parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
parser.add_argument('--warmup', type=float, default=2, help="Training epochs")
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--visdom', action='store_true',
                    help="Whether to log using visdom")
parser.add_argument('--ignore_checkpoints', action='store_true',
                    help='If specified, avoid loading existing models (no pre-training)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--load_fusion_weight', action='store_true')
parser.add_argument('--alpha_mu', type=float, default=1.0)
parser.add_argument('--alpha_dropout', type=float, default=0.0)
parser.add_argument('--mlp_random_init', action='store_true')
parser.add_argument('--mlp_lr_mul', type=float, default=1.0)
parser.add_argument('--weight_late_fusion', type=float, default=1.0)
parser.add_argument('--data_id', type=str, default='1')

mean = {'1':[2.1003, 0.7547, 0.0026], '2': [2.8940, 0.7540, 0.0026], '3': [2.4246, 0.7554, 0.0026]}
std = {'1':[5.3426, 0.3261, 0.0368], '2': [7.8383, 0.3263, 0.0370], '3': [6.4988, 0.3259, 0.0369]}
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
    path_to_lmdb = [join(args.path_to_data, "rgb")]
    path_to_lmdb.append(join(args.path_to_data, 'flow'))
    path_to_lmdb.append(join(args.path_to_data, 'obj'))
    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, f"{mode}{args.data_id}.csv"),
        'time_step': args.alpha,
        'img_tmpl': args.img_tmpl,
        'action_samples': args.S_ant if args.task == 'early_recognition' else None,
        'past_features': args.task == 'anticipation',
        'sequence_length': args.S_enc + args.S_ant,
        'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
        'challenge': 'test' in mode,
        'split_num': args.data_id
    }

    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training')


def get_model(mode):
    assert mode in ('pre', 'fut')
    rgb_model = SingleTransformer(args.feats_in[0], args.hidden, args.nheads, args.nlayers, args.dropout)
    rgb_classifier = torch.nn.Sequential(
        torch.nn.Dropout(args.fusion_dropout),
        torch.nn.Linear(args.hidden, args.num_class)
    )
    flow_model = SingleTransformer(args.feats_in[1], args.hidden, args.nheads, args.nlayers, args.dropout)
    flow_classifier = torch.nn.Sequential(
        torch.nn.Dropout(args.fusion_dropout),
        torch.nn.Linear(args.hidden, args.num_class)
    )
    obj_model = SingleTransformer(args.feats_in[2], args.hidden, args.nheads, args.nlayers, args.dropout)
    obj_classifier = torch.nn.Sequential(
        torch.nn.Dropout(args.fusion_dropout),
        torch.nn.Linear(args.hidden, args.num_class)
    )
    fusion_model = FusionLinear(args.modality_in, args.hidden, args.feat_out, args.num_class, args.nheads, args.nlayers, args.fusion_dropout)

    logger.info(rgb_model.load_state_dict(torch.load(
        os.path.join(args.path_to_models, f'Transformer-early_recognition_0.25_8_{mode}_rgb_best.pth.tar'))['state_dict']))
    logger.info(rgb_classifier.load_state_dict(torch.load(
        os.path.join(args.path_to_models, f'Transformer-early_recognition_0.25_8_{mode}_rgb_classifier_best.pth.tar'))['state_dict']))

    logger.info(flow_model.load_state_dict(torch.load(
        os.path.join(args.path_to_models, f'Transformer-early_recognition_0.25_8_{mode}_flow_best.pth.tar'))['state_dict']))
    logger.info(flow_classifier.load_state_dict(torch.load(
        os.path.join(args.path_to_models, f'Transformer-early_recognition_0.25_8_{mode}_flow_classifier_best.pth.tar'))['state_dict']))

    logger.info(obj_model.load_state_dict(torch.load(
        os.path.join(args.path_to_models, f'Transformer-early_recognition_0.25_8_{mode}_obj_best.pth.tar'))['state_dict']))
    logger.info(obj_classifier.load_state_dict(torch.load(
        os.path.join(args.path_to_models, f'Transformer-early_recognition_0.25_8_{mode}_obj_classifier_best.pth.tar'))['state_dict']))

    if args.load_fusion_weight:
        logger.info(fusion_model.load_state_dict(torch.load(
            os.path.join(args.path_to_models, f'Transformer-early_recognition_0.25_8_{mode}_fusion_best.pth.tar'))['state_dict']))

    return [rgb_model, flow_model, obj_model], [rgb_classifier, flow_classifier, obj_classifier], fusion_model


def load_checkpoint(model, dirname, best=False, model_name='pre_fusion'):
    if best:
        chk = torch.load(join(dirname, exp_name + '_' + model_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(dirname, exp_name + '_' + model_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(model, epoch, perf, best_perf, dirname, is_best=False, model_name='pre_fusion'):
    # pass
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(dirname, exp_name + '_' + model_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            dirname, exp_name + '_' + model_name + '_best.pth.tar'))
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


def log(mode, epoch, loss_meter, accuracy_pre_meter, accuracy_pre_meter_ens, best_perf=None, green=False):
    if green:
        raise NotImplementedError
        # logger.info('\033[92m', end="")
    msg = f"[{mode}] Epoch: {epoch:0.2f}. Loss: {loss_meter.value():.2f}. "
    acc_pre = 0
    acc_pre_ens = 0
    msg_pre = ''
    msg_pre_ens = ''
    for k in range(8):
        acc_k_pre = accuracy_pre_meter[k].value()
        msg_pre += f'{12.5*(k+1):.1f}%: {acc_k_pre:.2f}%, '
        acc_pre += acc_k_pre

        acc_k_pre_ens = accuracy_pre_meter_ens[k].value()
        msg_pre_ens += f'{12.5 * (k + 1):.1f}%: {acc_k_pre_ens:.2f}%, '
        acc_pre_ens += acc_k_pre_ens

    acc_pre = acc_pre/8
    acc_pre_ens = acc_pre_ens/8

    if best_perf:
        logger.info(f"{msg}[PRE:{msg_pre}]Avg: {acc_pre: .2f}%[best: {best_perf:0.2f}%]")
        logger.info(f"{msg}[PRE:{msg_pre_ens}]Avg: {acc_pre_ens: .2f}%")
    else:
        logger.info(f"{msg}[PRE:{msg_pre}]Avg: {acc_pre: .2f}%")
        logger.info(f"{msg}[PRE:{msg_pre_ens}]Avg: {acc_pre_ens: .2f}%")

def get_scores(pre_models, pre_fusion_model, loader, k):
    pre_fusion_model.eval()
    for i in range(args.modality_in):
        pre_models[i].eval()
    predictions = []
    labels = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['action_features']
            x = [xx.to(device) for xx in x]
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
        pre_models, pre_fusion_model, fut_models, fut_fusion_model, mlp, classifiers, loaders, optimizer, epochs, scheduler=None):
    """Training/Validation code"""
    best_perf = 0
    best_perf_modalities = [0, 0, 0]
    save_dir = os.path.join(args.path_to_models, args.path_to_models_stage2)
    for epoch in range(epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_pre_meter = {'training': [ValueMeter() for _ in range(8)], 'validation': [ValueMeter() for _ in range(8)]}
        accuracy_pre_meter_ens = {'training': [ValueMeter() for _ in range(8)], 'validation': [ValueMeter() for _ in range(8)]}
        for mode in ['validation', 'training']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    pre_fusion_model.train()
                    fut_fusion_model.train()
                    mlp.train()
                    for i in range(args.modality_in):
                        classifiers[i].train()
                    for i, batch in enumerate(loaders[mode]):
                        x = batch['action_features']
                        x[0] = (x[0] - mean[args.data_id][0]) / std[args.data_id][0]
                        x[1] = (x[1] - mean[args.data_id][1]) / std[args.data_id][1]
                        x[2] = (x[2] - mean[args.data_id][2]) / std[args.data_id][2]
                        x = [xx.to(device) for xx in x]

                        # 先训练一边pre-model
                        losses_pre = []
                        for k in range(1, 9):
                            x1 = [xx[:, :k, :] for xx in x]
                            y = batch['label'].to(device)

                            bs = y.shape[0]  # batch size

                            # pre_model计算
                            pre_features = [pre_models[j](x1[j]) for j in range(args.modality_in)]
                            apre_rgb = classifiers[0](pre_features[0])
                            apre_flow = classifiers[1](pre_features[1])
                            apre_obj = classifiers[2](pre_features[2])

                            pre_features = torch.stack(pre_features, 1)

                            app, hpp, hpf = pre_fusion_model(pre_features, 'pre')
                            apf = fut_fusion_model.fc(hpf)
                            weight_app = mlp(args.alpha_mu * torch.cat((hpp, hpf), dim=-1))  # 权重要不要加dropout

                            # apre = 0.5 * (app + apf)
                            apre = weight_app * app + (1 - weight_app) * apf
                            loss_p = F.cross_entropy(apre, y) + args.weight_late_fusion * (
                                     0.33 * F.cross_entropy(apre_rgb, y) +
                                     0.33 * F.cross_entropy(apre_flow, y) +
                                     0.33 * F.cross_entropy(apre_obj, y)
                            )

                            apre_ens = 0.5 * apre + 0.5 * 0.33 * (apre_rgb + apre_flow + apre_obj)

                            losses_pre.append(loss_p)

                            acc_pre = topk_accuracy(
                                apre.detach().cpu().numpy(), y.detach().cpu().numpy(), (1,))[0]*100
                            acc_pre_ens = topk_accuracy(
                                apre_ens.detach().cpu().numpy(), y.detach().cpu().numpy(), (1,))[0]*100
                            accuracy_pre_meter[mode][k-1].add(acc_pre, bs)
                            accuracy_pre_meter_ens[mode][k-1].add(acc_pre_ens, bs)
                        # pdb.set_trace()
                        loss_pre = sum(losses_pre)/8

                        optimizer.zero_grad()
                        loss_pre.backward()
                        optimizer.step()

                        loss_meter[mode].add(loss_pre.item(), bs)

                        e = epoch + i/len(loaders[mode])

                        if i != 0 and i % args.display_every == 0:
                            log(mode, e, loss_meter[mode], accuracy_pre_meter[mode], accuracy_pre_meter_ens[mode])
                else:
                    sum_accuracy_pre = 0
                    sum_accuracy_pre_rgb = 0
                    sum_accuracy_pre_flow = 0
                    sum_accuracy_pre_obj = 0
                    sum_accuracy_pre_ens = 0
                    pre_fusion_model.eval()
                    fut_fusion_model.eval()
                    mlp.eval()
                    for i in range(args.modality_in):
                        classifiers[i].eval()
                    for k in range(0, 8):
                        predictions_pre = []
                        predictions_pre_rgb = []
                        predictions_pre_flow = []
                        predictions_pre_obj = []
                        predictions_pre_ens = []
                        labels = []
                        for i, batch in enumerate(loaders[mode]):
                            x = batch['action_features']
                            x[0] = (x[0] - mean[args.data_id][0]) / std[args.data_id][0]
                            x[1] = (x[1] - mean[args.data_id][1]) / std[args.data_id][1]
                            x[2] = (x[2] - mean[args.data_id][2]) / std[args.data_id][2]
                            x = [xx.to(device) for xx in x]
                            x1 = [xx[:, :k+1, :] for xx in x]
                            y = batch['label'].numpy()

                            pre_features = [pre_models[j](x1[j]) for j in range(args.modality_in)]
                            apre_rgb = classifiers[0](pre_features[0])
                            apre_flow = classifiers[1](pre_features[1])
                            apre_obj = classifiers[2](pre_features[2])

                            pre_features = torch.stack(pre_features, 1)

                            app, hpp, hpf = pre_fusion_model(pre_features, 'pre')
                            apf = fut_fusion_model.fc(hpf)
                            weight_app = mlp(args.alpha_mu * torch.cat((hpp, hpf), dim=-1))  # 权重要不要加dropout
                            apre = weight_app * app + (1 - weight_app) * apf
                            apre_ens = 0.5 * apre + 0.5 * 0.33 * (apre_rgb + apre_flow + apre_obj)
                            apre = apre.cpu().numpy()
                            predictions_pre.append(apre)
                            apre_ens = apre_ens.cpu().numpy()
                            predictions_pre_ens.append(apre_ens)

                            predictions_pre_rgb.append(apre_rgb.cpu().numpy())
                            predictions_pre_flow.append(apre_flow.cpu().numpy())
                            predictions_pre_obj.append(apre_obj.cpu().numpy())

                            labels.append(y)


                        action_scores_pre = np.concatenate(predictions_pre)
                        action_scores_pre_ens = np.concatenate(predictions_pre_ens)

                        action_scores_pre_rgb = np.concatenate(predictions_pre_rgb)
                        action_scores_pre_flow = np.concatenate(predictions_pre_flow)
                        action_scores_pre_obj = np.concatenate(predictions_pre_obj)

                        labels = np.concatenate(labels)
                        acc_pre = topk_accuracy(action_scores_pre, labels, (1,))[0]*100

                        acc_pre_rgb = topk_accuracy(action_scores_pre_rgb, labels, (1,))[0]*100
                        acc_pre_flow = topk_accuracy(action_scores_pre_flow, labels, (1,))[0]*100
                        acc_pre_obj = topk_accuracy(action_scores_pre_obj, labels, (1,))[0]*100

                        acc_pre_ens = topk_accuracy(action_scores_pre_ens, labels, (1,))[0]*100
                        sum_accuracy_pre += acc_pre

                        sum_accuracy_pre_rgb += acc_pre_rgb
                        sum_accuracy_pre_flow += acc_pre_flow
                        sum_accuracy_pre_obj += acc_pre_obj

                        sum_accuracy_pre_ens += acc_pre_ens
                        logger.info('\033[92m'
                                      f"Epoch: {epoch + 1:0.2f}. Ratio: {float(k+1)/8*100:0.1f}%. "
                                      f"early fusion: {acc_pre:0.2f}%. "
                                      f"rgb: {acc_pre_rgb:0.2f}%. "
                                      f"flow: {acc_pre_flow:0.2f}%. "
                                      f"obj: {acc_pre_obj:0.2f}%. "
                                      f"ensemble: {acc_pre_ens:0.2f}%. "
                                      '\033[0m')
                        if k == 7:
                            avg_accuracy_pre = sum_accuracy_pre / 8
                            avg_accuracy_pre_rgb = sum_accuracy_pre_rgb / 8
                            avg_accuracy_pre_flow = sum_accuracy_pre_flow / 8
                            avg_accuracy_pre_obj = sum_accuracy_pre_obj / 8
                            avg_accuracy_pre_ens = sum_accuracy_pre_ens / 8
                            avg_accuracy_pre_modalities = [
                                avg_accuracy_pre_rgb, avg_accuracy_pre_flow, avg_accuracy_pre_obj]
                            logger.info('\033[92m'
                                          f"[{mode}] Epoch: {epoch + 1:0.2f}. "
                                          f"early fusion: {avg_accuracy_pre:.2f}%. "
                                          f"rgb: {avg_accuracy_pre_rgb:.2f}%. "
                                          f"flow: {avg_accuracy_pre_flow:.2f}%. "
                                          f"obj: {avg_accuracy_pre_obj:.2f}%. "
                                          f"ensemble: {avg_accuracy_pre_ens:.2f}%. "
                                          f"[best: {max(avg_accuracy_pre_ens, best_perf):0.2f}]%"
                                          '\033[0m')

                            if best_perf < avg_accuracy_pre_ens:
                                best_perf = avg_accuracy_pre_ens
                                is_best = True
                            else:
                                is_best = False

                            save_model(pre_fusion_model, epoch+1, avg_accuracy_pre, best_perf, dirname=save_dir, is_best=is_best, model_name=f'pre_fusion')
                            save_model(fut_fusion_model, epoch+1, avg_accuracy_pre, best_perf, dirname=save_dir, is_best=is_best, model_name=f'fut_fusion')
                            save_model(mlp, epoch+1, avg_accuracy_pre, best_perf, dirname=save_dir, is_best=is_best, model_name=f'mlp')
                            for j, modal in enumerate(['rgb', 'flow', 'obj']):
                                if best_perf_modalities[j] < avg_accuracy_pre_modalities[j]:
                                    best_perf_modalities[j] = avg_accuracy_pre_modalities[j]
                                    is_best = True
                                else:
                                    is_best = False
                                save_model(
                                    classifiers[j], epoch+1, avg_accuracy_pre_modalities[j], best_perf_modalities[j],
                                    dirname=save_dir, is_best=is_best, model_name=f'classifier_{modal}')
                    # if mode == 'training':

        if scheduler is not None:
            scheduler.step()

    logger.info(f'Final Validation......epochs={args.epochs}')
    load_checkpoint(pre_fusion_model, save_dir, True, 'pre_fusion')
    pre_fusion_model.eval()
    load_checkpoint(fut_fusion_model, save_dir, True, 'fut_fusion')
    fut_fusion_model.eval()
    load_checkpoint(mlp, save_dir, True, 'mlp')
    mlp.eval()
    for j, modal in enumerate(['rgb', 'flow', 'obj']):
        load_checkpoint(classifiers[j], save_dir, True, f'classifier_{modal}')
        classifiers[j].eval()
    predictions = []
    with torch.set_grad_enabled(False):
        for k in range(0, 8):
            predictions_pre_rgb = []
            predictions_pre_flow = []
            predictions_pre_obj = []
            predictions_pre_ens = []
            labels = []
            for i, batch in enumerate(loaders['validation']):
                x = batch['action_features']
                x[0] = (x[0] - mean[args.data_id][0]) / std[args.data_id][0]
                x[1] = (x[1] - mean[args.data_id][1]) / std[args.data_id][1]
                x[2] = (x[2] - mean[args.data_id][2]) / std[args.data_id][2]

                x = [xx.to(device) for xx in x]
                x1 = [xx[:, :k + 1, :] for xx in x]
                y = batch['label'].numpy()

                pre_features = [pre_models[j](x1[j]) for j in range(args.modality_in)]
                apre_rgb = classifiers[0](pre_features[0])
                apre_flow = classifiers[1](pre_features[1])
                apre_obj = classifiers[2](pre_features[2])

                pre_features = torch.stack(pre_features, 1)

                app, hpp, hpf = pre_fusion_model(pre_features, 'pre')
                apf = fut_fusion_model.fc(hpf)
                weight_app = mlp(args.alpha_mu * torch.cat((hpp, hpf), dim=-1))  # 权重要不要加dropout
                apre = weight_app * app + (1 - weight_app) * apf
                apre_ens = 0.5 * apre + 0.5 * 0.33 * (apre_rgb + apre_flow + apre_obj)
                apre_ens = apre_ens.cpu().numpy()
                predictions_pre_ens.append(apre_ens)

                predictions_pre_rgb.append(apre_rgb.cpu().numpy())
                predictions_pre_flow.append(apre_flow.cpu().numpy())
                predictions_pre_obj.append(apre_obj.cpu().numpy())

                labels.append(y)

            action_scores_pre_ens = np.concatenate(predictions_pre_ens)
            action_scores_pre_rgb = np.concatenate(predictions_pre_rgb)
            action_scores_pre_flow = np.concatenate(predictions_pre_flow)
            action_scores_pre_obj = np.concatenate(predictions_pre_obj)

            labels = np.concatenate(labels)
            predictions.append({
                'ens': action_scores_pre_ens,
                'rgb': action_scores_pre_rgb,
                'flow': action_scores_pre_flow,
                'obj': action_scores_pre_obj,
                'labels': labels
            })
                
    accs = []
    x = np.arange(0.01, 0.4, 0.01)
    for wrgb in (0, 0.12):
        for wflow in (0, 0.27):
            for wobj in (0, 0.01):
                acc = 0
                wearly = 1 - wrgb - wflow - wobj
                acc_perk = []
                for k in range(8):
                    action_scores = wearly * predictions[k]['ens'] + wrgb * predictions[k]['rgb'] +\
                                    wflow * predictions[k]['flow'] + wobj * predictions[k]['obj']
                    labels = predictions[k]['labels']
                    acc += (action_scores.argmax(-1) == labels).mean()
                    acc_perk.append((action_scores.argmax(-1) == labels).mean())
                acc = acc / 8
                #logger.info(f'{wearly:.2f}, {wrgb:.2f}, {wflow:.2f}, {wobj:.2f}, {acc:.2f}')
                accs.append((acc, wrgb, wflow, wobj, acc_perk))
    best_perf, wrgb, wflow, wobj, acc_perk = max(accs)
    wearly = 1 - wrgb - wflow - wobj
    logger.info(f'\033[92m Final Validation Avg_Pre_Accuracy: {best_perf * 100:.2f}%'
                f'({wearly:.2f}, {wrgb:.2f}, {wflow:.2f}, {wobj:.2f}).\033[0m')
    for k in range(8):
        logger.info(f'\033[92m Final Validation Avg_Pre_Accuracy: {acc_perk[k] * 100:.2f}%')


def cosine_annealing_with_warmup(warm_up_iter, max_iter, lr_min, lr_max):
    def fun(cur_iter):
        if cur_iter < warm_up_iter:
            return (cur_iter+1)/warm_up_iter
        else:
            return (lr_min + 0.5 * (lr_max - lr_min)*(1. + math.cos(math.pi * (cur_iter - warm_up_iter)/(max_iter-warm_up_iter))))/lr_max
    return fun

def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 40], 0.5)


def main():
    set_seed(args.seed)
    # 新建目录
    if not os.path.exists(os.path.join(args.path_to_models, args.path_to_models_stage2)):
        os.makedirs(os.path.join(args.path_to_models, args.path_to_models_stage2))
    global logger
    logger = logging.getLogger(f'main{random.randint(0, 1e5)}')
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(logging.FileHandler(os.path.join(args.path_to_models, args.path_to_models_stage2, 'log.log')))
    pre_models, classifiers, pre_fusion_model = get_model('pre')
    fut_models, _, fut_fusion_model = get_model('fut')

    mlp = torch.nn.Sequential(
        torch.nn.Dropout(args.fusion_dropout),
        torch.nn.Linear(args.feat_out * 2, 1),
        torch.nn.Dropout(args.alpha_dropout),
        torch.nn.Sigmoid(),
    )

    if not args.mlp_random_init:
        torch.nn.init.zeros_(mlp[1].weight)
        torch.nn.init.zeros_(mlp[1].bias)

    for i in range(args.modality_in):
        pre_models[i].requires_grad_(False)
        fut_models[i].requires_grad_(False)
        pre_models[i].eval()
        fut_models[i].eval()

    pre_models = [m.to(device) for m in pre_models]
    fut_models = [m.to(device) for m in fut_models]
    classifiers = [m.to(device) for m in classifiers]
    pre_fusion_model.to(device)
    fut_fusion_model.to(device)
    mlp.to(device)
    
    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}
        optimizer = torch.optim.AdamW([
            {"params": pre_fusion_model.parameters()},
            {"params": fut_fusion_model.parameters()},
            {"params": classifiers[0].parameters()},
            {"params": classifiers[1].parameters()},
            {"params": classifiers[2].parameters()},
            {"params": mlp.parameters(), "lr": args.lr * args.mlp_lr_mul, "weight_decay": 0.}
        ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))

        if args.lr_scheduler:
            scheduler = get_scheduler(optimizer)
            trainval(pre_models, pre_fusion_model, fut_models, fut_fusion_model, mlp, classifiers, loaders, optimizer, args.epochs, scheduler)
        else:
            trainval(pre_models, pre_fusion_model, fut_models, fut_fusion_model, mlp, classifiers, loaders, optimizer, args.epochs)


if __name__ == '__main__':
    main()