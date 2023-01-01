"""Main training/test program for RULSTM"""
import itertools
import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random

from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
from models import SingleTransformerPretrain
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
parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
parser.add_argument('--num_workers', type=int, default=4,
                    help="Number of parallel thread to fetch the data")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=1e-6, help="Minimum learning rate")
parser.add_argument('--weight_decay', type=float, default=0, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
parser.add_argument('--patience', type=int, default=20, help="early stop if performance do not increase after "
                                                             "'patience' epochs")
parser.add_argument('--display_every', type=int, default=10,
                    help="Display every n iterations")
parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
parser.add_argument('--warmup', type=int, default=10, help="number of warmup epochs")
parser.add_argument('--visdom', action='store_true',
                    help="Whether to log using visdom")
parser.add_argument('--ignore_checkpoints', action='store_true',
                    help='If specified, avoid loading existing models (no pre-training)')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume suspended training')
parser.add_argument('--ek100', action='store_true',
                    help="Whether to use EPIC-KITCHENS-100")
parser.add_argument('--lbd', default=1.0, help="mse loss weight")
parser.add_argument('--seed', default=42, help='random seed')

parser.add_argument('--json_directory', type=str, default = None, help = 'Directory in which to save the generated jsons.')
parser.add_argument('--data_id', type=str, default='1')

args = parser.parse_args()

mean = {'1':[2.1003, 0.7547, 0.0026], '2': [2.8940, 0.7540, 0.0026], '3': [2.4246, 0.7554, 0.0026]}
std = {'1':[5.3426, 0.3261, 0.0368], '2': [7.8383, 0.3263, 0.0370], '3': [6.4988, 0.3259, 0.0369]}

if args.mode == 'test' or args.mode=='validate_json':
    assert args.json_directory is not None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

exp_name = 'transformer_pretrain'



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
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


def load_checkpoint(model, best=False):
    if best:
        chk = torch.load(join(args.path_to_models, exp_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(args.path_to_models, exp_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(model, modality, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.path_to_models, f'{exp_name}_{modality}_{args.data_id}_last.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.path_to_models, f'{exp_name}_{modality}_{args.data_id}_best.pth.tar'))


def log(mode, epoch, loss_meters, perfs=None, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")
    time_steps = len(loss_meters['obj'])
    mess = f"[{mode}] Epoch: {epoch:0.2f}. " \
           f"Loss:\n"
    for modal in ('rgb', 'flow', 'obj'):
        mess += modal.ljust(6, ' ')
        for t in range(time_steps):
            mess += f"{loss_meters[modal][t].value():.2f} "
        mess += '\n'
    if perfs is None:
        print(mess, end="")
    else:
        print(mess)
        print("Loss: ", end='')
        for modal in ['rgb', 'flow', 'obj']:
            print(f"{perfs[modal][-1]: 0.2f}% ", end="")
            if best_perf:
                print(f"[best: {best_perf[modal]:0.2f}%] ", end="")

    print('\033[0m')


def trainval(models, loaders, optimizers, epochs, start_epoch, start_best_perf, schedulers=None):
    """Training/Validation code"""
    # best_perf = start_best_perf  # to keep track of the best performing epoch
    # perfs = []
    best_perf = {'rgb': math.inf, 'flow': math.inf, 'obj': math.inf}
    perfs = {'rgb': [], 'flow': [], 'obj': []}
    for epoch in range(start_epoch, epochs):
        loss_meter = {
            'training': {
                'rgb': [ValueMeter() for _ in range(1, 8)],
                'flow': [ValueMeter() for _ in range(1, 8)],
                'obj': [ValueMeter() for _ in range(1, 8)]
            },
            'validation': {
                'rgb': [ValueMeter() for _ in range(1, 8)],
                'flow': [ValueMeter() for _ in range(1, 8)],
                'obj': [ValueMeter() for _ in range(1, 8)]
            }
        }
        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    for i in range(args.modality_in):
                        models[i].train()
                else:
                    for i in range(args.modality_in):
                        models[i].eval()

                for i, batch in enumerate(loaders[mode]):
                    x = batch['action_features']
                    x[0] = (x[0] - mean[args.data_id][0]) / std[args.data_id][0]
                    x[1] = (x[1] - mean[args.data_id][1]) / std[args.data_id][1]
                    x[2] = (x[2] - mean[args.data_id][2]) / std[args.data_id][2]
                    if type(x) == list:
                        x = [xx.to(device) for xx in x]
                    else:
                        x = x.to(device)
                    y = batch['label'].to(device)
                    bs = y.shape[0]  # batch size

                    losses = {'rgb': [], 'flow': [], 'obj': []}
                    for t in range(1, 8):
                        for j, modal in enumerate(['rgb', 'flow', 'obj']):
                            features = models[j](x[j][:, :t])
                            loss = F.mse_loss(features, x[j][:, t])
                            losses[modal].append(loss)
                            loss_meter[mode][modal][t-1].add(loss.item(), bs)

                    # if in training mode
                    if mode == 'training':
                        for j, modal in enumerate(['rgb', 'flow', 'obj']):
                            optimizers[j].zero_grad()
                            loss = torch.stack(losses[modal]).mean()
                            loss.backward()
                        #clip_grad_norm_(fusion_model.parameters(), 2)

                            optimizers[j].step()

                    # compute decimal epoch for logging
                    e = epoch + i/len(loaders[mode])

                    # log training during loop
                    # avoid logging the very first batch. It can be biased.
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode])

        print(f"lr: {optimizers[0].param_groups[0]['lr']}.")
        for j, modal in enumerate(['rgb', 'flow', 'obj']):
            perf = np.mean([loss.value() for loss in loss_meter['validation'][modal]])
            perfs[modal].append(perf)
            if best_perf[modal] > perf:
                best_perf[modal] = perf
                save_model(models[j], modal, epoch, perfs[modal], best_perf[modal], is_best=True)
            else:
                save_model(models[j], modal, epoch, perfs[modal], best_perf[modal], is_best=False)
        log(mode, epoch + 1, loss_meter['validation'], perfs, best_perf, green=True)
        if schedulers is not None:
            for sch in schedulers:
                sch.step()
        # log at the end of each epoch
    return best_perf


def cosine_annealing_with_warmup(warm_up_iter, max_iter, lr_min, lr_max):
    def fun(cur_iter):
        if cur_iter < warm_up_iter:
            return (cur_iter+1)/warm_up_iter
        else:
            return (lr_min + 0.5 * (lr_max - lr_min)*(1. + math.cos(math.pi * (cur_iter - warm_up_iter)/(max_iter-warm_up_iter))))/lr_max
    return fun


def get_model():
    rgb_model = SingleTransformerPretrain(args.feats_in[0], args.hidden, args.nheads, args.nlayers, args.dropout)
    flow_model = SingleTransformerPretrain(args.feats_in[1], args.hidden, args.nheads, args.nlayers, args.dropout)
    obj_model = SingleTransformerPretrain(args.feats_in[2], args.hidden, args.nheads, args.nlayers, args.dropout)
    return rgb_model, flow_model, obj_model


def main():
    set_seed(args.seed)
    # 新建目录
    if not os.path.exists(args.path_to_models):
        os.makedirs(args.path_to_models)
    logging.basicConfig(filename=os.path.join(args.path_to_models, 'log.log'), level=logging.DEBUG)
    models = get_model()
    models = [model.to(device) for model in models]
    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}

        if args.resume:
            raise NotImplementedError
            # start_epoch, _, start_best_perf = load_checkpoint(fusion_model)
        else:
            start_epoch = 0
            start_best_perf = [math.inf for _ in range(3)]

        optimizers = [torch.optim.Adam(models[i].parameters(), lr=args.lr, betas=[0.5, 0.999],
                                       weight_decay=args.weight_decay) for i in range(args.modality_in)]
        schedulers = [torch.optim.lr_scheduler.LambdaLR(
            optimizers[i], lr_lambda=cosine_annealing_with_warmup(args.warmup, args.epochs, args.lr_min, args.lr)) for
        i in range(args.modality_in)]
        # scheduler = None

        res = trainval(models, loaders, optimizers, args.epochs, start_epoch, start_best_perf, schedulers)
    return res


if __name__ == '__main__':
    main()