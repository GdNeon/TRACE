"""Main training/test program for RULSTM"""
import itertools
import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import random

from argparse import ArgumentParser
from dataset import SequenceDataset
from os.path import join
from models import FusionTransformer, FusionLinear, RULSTM, SingleTransformer, SingleTransformerPretrain
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
parser.add_argument('--num_class', type=int, default=2513,
                    help='Number of classes')
parser.add_argument('--hidden', type=int, default=768,
                    help='Number of hidden units')
parser.add_argument('--feats_in', type=int, nargs='+', default=[1024, 1024, 352, 1024],
                    help='Input sizes when the fusion modality is selected.')
parser.add_argument('--modality_in', type=int, default=4)
parser.add_argument('--feat_out', type=int, default=1024)
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

args = parser.parse_args()

if args.mode == 'test' or args.mode=='validate_json':
    assert args.json_directory is not None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# if args.task == 'anticipation':
#     exp_name = f"Transformer-{args.task}_{args.alpha}_{args.S_enc}_{args.S_ant}"
# else:
#     exp_name = f"Transformer-{args.task}_{args.alpha}_{args.S_ant}"
#
# if args.sequence_completion:
#     exp_name += '_sequence_completion'
exp_name = 'transformer_pretrain'
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = None

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
    torch.backends.cudnn.enabled = False
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
        'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
        'challenge': 'test' in mode
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
                'perf': perf, 'best_perf': best_perf}, join(args.path_to_models, f'{exp_name}_{modality}_last.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.path_to_models, f'{exp_name}_{modality}_best.pth.tar'))

    if args.visdom:
        # save visdom logs for persitency
        visdom_saver.save()


def log(mode, epoch, loss_meters, perfs=None, best_perf=None, green=False):
    # if green:
    #     print('\033[92m', end="")
    time_steps = len(loss_meters['obj'])
    mess = f"[{mode}] Epoch: {epoch:0.2f}. " \
           f"Loss:\n"
    for modal in ('rgb', 'flow', 'obj', 'audio'):
        mess += modal.ljust(6, ' ')
        for t in range(time_steps):
            mess += f"{loss_meters[modal][t].value():.2f} "
        mess += '\n'
    if perfs is not None:
        mess += '\nLoss: '
        for modal in ['rgb', 'flow', 'obj', 'audio']:
            mess += f"{perfs[modal][-1]: 0.2f}% "
            if best_perf:
                mess += f"[best: {best_perf[modal]:0.2f}%] "
    logger.info(mess)

    # print('\033[0m')


def trainval(models, loaders, optimizers, epochs, start_epoch, start_best_perf, schedulers=None):
    """Training/Validation code"""
    # best_perf = start_best_perf  # to keep track of the best performing epoch
    # perfs = []
    best_perf = {'rgb': math.inf, 'flow': math.inf, 'obj': math.inf, 'audio': math.inf}
    perfs = {'rgb': [], 'flow': [], 'obj': [], 'audio': []}
    for epoch in range(start_epoch, epochs):
        # define training and validation meters
        # loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        # accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        loss_meter = {
            'training': {
                'rgb': [ValueMeter() for _ in range(1, 8)],
                'flow': [ValueMeter() for _ in range(1, 8)],
                'obj': [ValueMeter() for _ in range(1, 8)],
                'audio': [ValueMeter() for _ in range(1, 8)],
            },
            'validation': {
                'rgb': [ValueMeter() for _ in range(1, 8)],
                'flow': [ValueMeter() for _ in range(1, 8)],
                'obj': [ValueMeter() for _ in range(1, 8)],
                'audio': [ValueMeter() for _ in range(1, 8)],
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
                    x[0] = (x[0] - 0.786) / 1.532
                    x[1] = (x[1] - 1.033) / 2.666
                    x[2] = (x[2] - 0.006) / 0.065
                    x[3] = (x[3] - 1.086) / 2.642

                    if type(x) == list:
                        x = [xx.to(device) for xx in x]
                    else:
                        x = x.to(device)
                    # x = [lstm_models[i](x[i]) for i in range(args.modality_in)]
                    y = batch['label'].to(device)
                    bs = y.shape[0]  # batch size

                    losses = {'rgb': [], 'flow': [], 'obj': [], 'audio': []}
                    for t in range(1, 8):
                        for j, modal in enumerate(['rgb', 'flow', 'obj', 'audio']):
                            features = models[j](x[j][:, :t])
                            loss = F.mse_loss(features, x[j][:, t])
                            losses[modal].append(loss)
                            loss_meter[mode][modal][t-1].add(loss.item(), bs)

                    # if in training mode
                    if mode == 'training':
                        for j, modal in enumerate(['rgb', 'flow', 'obj', 'audio']):
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

        logger.info(f"lr: {optimizers[0].param_groups[0]['lr']}.")
        for j, modal in enumerate(['rgb', 'flow', 'obj', 'audio']):
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
    audio_model = SingleTransformerPretrain(args.feats_in[3], args.hidden, args.nheads, args.nlayers, args.dropout)
    return rgb_model, flow_model, obj_model, audio_model


def main():
    set_seed(args.seed)
    # 新建目录
    print(args.path_to_models)
    if not os.path.exists(args.path_to_models):
        print(args.path_to_models)
        os.makedirs(args.path_to_models)
    global logger
    logger = logging.getLogger(f'main{random.randint(0, 1e5)}')
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    # logger.removeHandler(logger.handlers)
    logger.addHandler(logging.FileHandler(os.path.join(args.path_to_models, 'log.log')))
    # model = UniterPretrain(
    #     args.feats_in, args.hidden, args.nheads, args.nlayers, args.dropout).to(device)
    models = get_model()
    models = [model.to(device) for model in models]
    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}

        if args.resume:
            raise NotImplementedError
            # start_epoch, _, start_best_perf = load_checkpoint(fusion_model)
        else:
            start_epoch = 0
            start_best_perf = [math.inf for _ in range(4)]

        optimizers = [torch.optim.Adam(models[i].parameters(), lr=args.lr, betas=[0.5, 0.999],
                                       weight_decay=args.weight_decay) for i in range(args.modality_in)]
        schedulers = [torch.optim.lr_scheduler.LambdaLR(
            optimizers[i], lr_lambda=cosine_annealing_with_warmup(args.warmup, args.epochs, args.lr_min, args.lr)) for
        i in range(args.modality_in)]
        # scheduler = None

        res = trainval(models, loaders, optimizers, args.epochs, start_epoch, start_best_perf, schedulers)
    return res


if __name__ == '__main__':
    # main()
    # for layer in (1, 2, 3, 6):
    #     for head in (1, 2, 4, 8):
    #         if layer == 3 and head == 4:
    #             continue
    for layer in (5, ):
        args.nlayers = layer
        args.nheads = 8
        args.path_to_models = f'/disk/wkj/Video-Transformer/models/transformer_pretrain1210/{args.nlayers}layer_{args.nheads}head'
        main()

    # for head in (32, 64, 128, 256):
    #     args.nlayers = 2
    #     args.nheads = head
    #     args.path_to_models = f'/disk/wkj/Video-Transformer/models/transformer_pretrain1210/{args.nlayers}layer_{args.nheads}head'
    #     main()
    # df = []
    # for head in (1, 2, 4, 8, 16):
    #     print(f'head: {head}')
    #     args.nheads = head
    #     acc = main()
    #     df.append((head, acc))
    # df = pd.DataFrame(df, columns=['head', 'acc'])
    # for layer in (12, 6, 3, 1):
    #     print(f'layer: {layer}')
    #     args.nlayers = layer
    #     acc = main()
    #     df.append((layer, acc))
    # df = pd.DataFrame(df, columns=['layer', 'acc'])
    # for lbd in (0.01, 0.1, 1, 10):
    #     print(f'lambda: {lbd}')
    #     args.lbd = lbd
    #     acc = main()
    #     df.append((lbd, acc))
    # df = pd.DataFrame(df, columns=['lambda', 'acc'])
    # print(df)
    # for lbd in (0.01, 0.1, 1, 10):
    #     print(f'lambda: {lbd}')
    #     args.lbd = lbd
    #     acc = main()
    #     df.append((lbd, acc))
    # df = pd.DataFrame(df, columns=['lambda', 'acc'])
    # for hidden in (256, 512, 768, 1024, 2048):
    #     print(f'hidden: {hidden}')
    #     args.hidden = hidden
    #     acc = main()
    #     df.append((hidden, acc))
    # df = pd.DataFrame(df, columns=['hidden', 'acc'])
    # print(df)
