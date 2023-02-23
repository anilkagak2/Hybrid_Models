#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)

Hybrid Models added by Anil Kag (https://anilkagak2.github.io)
"""
import argparse
import time
import math
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from routing_models import get_routing_model
from hybrid_datasets import hybrid_create_dataset
from hybrid_dataloaders import hybrid_create_loader
from hybrid_checkpoint_saver import HybridCheckpointSaver
from hybrid_models import get_model_from_name

from utils import count_net_flops
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

def get_model_prefix( args ):
    prefix =  args.model_type + '-' + args.model + '-' \
            + args.global_type + '-' + args.global_model + '-' \
            + args.routing_model + '-trn-' \
            + str(args.n_parts)+'-' 
    return prefix

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')

parser.add_argument('--wpos', type=float, default=3., help='target 1-coverage term in the formulation.')
parser.add_argument('--wneg', type=float, default=1., help='target 1-coverage term in the formulation.')

parser.add_argument('--n_parts', type=int, default=1, help='input batch size for training')
parser.add_argument('--cov', type=float, default=0.55, help='target 1-coverage term in the formulation.')
parser.add_argument('--cov2', type=float, default=0.65, help='target 1-coverage term in the formulation.')
parser.add_argument('--g_denom', type=float, default=1., help='denominator in the balanced routing loss formulation.')
parser.add_argument('--strategy', type=int, default=1, help='number of epochs per alternates')
parser.add_argument('--s_iters', type=int, default=510, help='batches for base optimization')
parser.add_argument('--t_iters', type=int, default=510, help='batches for global optimization')
parser.add_argument('--g_iters', type=int, default=110, help='batches for gate optimization')

parser.add_argument('--routing-model', default='no_ft', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--routing-ema-decay', type=float, default=0.99,
                    help='decay factor for model weights moving average (default: 0.9998)')

parser.add_argument('--global-type', default='ofa', type=str)
parser.add_argument('--global-model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')

parser.add_argument('--model-type', default='mcunet', type=str)
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')

parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--g-lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    assert( args.cov <= args.cov2 )

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def get_flops_params( model, resolution, label='Base' ):
    total_ops = count_net_flops(model, [1, 3, resolution, resolution])
    total_params = sum([p.numel() for p in model.parameters()])
    print(' * ['+label+'] FLOPs: {:.4}M, param: {:.4}M'.format(total_ops / 1e6, total_params / 1e6))
    x, ft = model( torch.randn(1,3, resolution, resolution) )
    print('Dummy output model --> x ft', x.size(), ft.size())
    num_ft = ft.size()[-1]
    return total_ops, total_params, num_ft

def get_model_with_stats( args, model_name, model_stats, label, model_type ):
    model, base_model_cfg = get_model_from_name( model_name, model_type, args )
    resolution = base_model_cfg['input_size'][2]

    total_ops, total_params, num_ft = get_flops_params( model, resolution, label )
    model_stats[model_name+'param'] = total_params
    model_stats[model_name+'flop'] = total_ops

    del model
    model, base_model_cfg = get_model_from_name( model_name, model_type, args )
    base_model_cfg['num_ft'] = num_ft
    return model, base_model_cfg

def main():
    setup_default_logging()
    args, args_text = _parse_args()
    
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else: 
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
             
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    part_iters = {}
    part_iters['gating'] = args.g_iters
    part_iters['student'] = args.s_iters
    part_iters['teacher'] = args.t_iters


    base_model_stats = {}
    global_model_stats = {}
    hybrid_model_stats = {}

    model, base_cfg = get_model_with_stats( args, args.model, base_model_stats, 'Base', args.model_type )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    routing_model = get_routing_model( routing_name=args.routing_model, base_model_cfg=base_cfg, )

    global_model, global_cfg = get_model_with_stats( args, args.global_model, global_model_stats, 'Global', args.global_type )
    assert( model.num_classes == global_model.num_classes )

    args.base_flops   = base_model_stats[ args.model+'flop' ] /1e6 
    args.global_flops = global_model_stats[ args.global_model+'flop' ] /1e6
    if args.local_rank == 0:
        _logger.info(
            f'GlobalModel {safe_model_name(args.global_model)} created, param count:{sum([m.numel() for m in global_model.parameters()])}')
    #assert(1==2)

    #data_config        = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    #global_data_config = resolve_data_config(vars(args), model=global_model, verbose=args.local_rank == 0)
    data_config = base_cfg
    global_data_config = global_cfg
    if args.local_rank == 0:
        _logger.info('Data processing configuration for :' + args.model)
        for n, v in data_config.items():
            _logger.info('\t%s: %s' % (n, str(v)))

        _logger.info('Data processing configuration for :' + args.global_model)
        for n, v in global_data_config.items():
            _logger.info('\t%s: %s' % (n, str(v)))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
        global_model = convert_splitbn_model(global_model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    routing_model.cuda()
    global_model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        global_model = global_model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
            global_model = convert_syncbn_model(global_model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            global_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(global_model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
        global_model = torch.jit.script(global_model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    global_optimizer = create_optimizer_v2(global_model, **optimizer_kwargs(cfg=args))
    s_lr = args.lr
    g_lr = args.g_lr
    args.lr = g_lr
    routing_optimizer = create_optimizer_v2(routing_model, **optimizer_kwargs(cfg=args))
    args.lr = s_lr

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        global_model, global_optimizer = amp.initialize(global_model, global_optimizer, opt_level='O1')
        routing_model, routing_optimizer = amp.initialize(routing_model, routing_optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    global_model_ema = None
    routing_model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        global_model_ema = ModelEmaV2(global_model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        routing_model_ema = ModelEmaV2(routing_model, decay=args.routing_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
            load_checkpoint(global_model_ema.module, args.resume_global, use_ema=True)
            load_checkpoint(routing_model_ema.module, args.resume_global, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
            global_model = ApexDDP(global_model, delay_allreduce=True)
            routing_model = ApexDDP(routing_model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
            global_model = NativeDDP(global_model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
            routing_model = NativeDDP(routing_model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    global_lr_scheduler, g_num_epochs = create_scheduler(args, global_optimizer)
    args.lr = g_lr
    routing_lr_scheduler, r_num_epochs = create_scheduler(args, routing_optimizer)
    assert(num_epochs == g_num_epochs)
    assert(num_epochs == r_num_epochs)

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    if global_lr_scheduler is not None and start_epoch > 0:
        global_lr_scheduler.step(start_epoch)
    if routing_lr_scheduler is not None and start_epoch > 0:
        routing_lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    '''dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size, repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)'''

    dataset_train = hybrid_create_dataset(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size, repeats=args.epoch_repeats)
    dataset_eval = hybrid_create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)


    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    assert(mixup_active == False)


    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    '''loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )'''

    loader_train = hybrid_create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        g_input_size=global_data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,

        g_interpolation=global_data_config['interpolation'], # 'bilinear',
        g_mean=global_data_config['mean'],  #IMAGENET_DEFAULT_MEAN,
        g_std=global_data_config['std'],  #IMAGENET_DEFAULT_STD,
        g_crop_pct=global_data_config['crop_pct'], #None,
    )

    loader_eval = hybrid_create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        g_input_size=global_data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,

        g_interpolation=global_data_config['interpolation'], # 'bilinear',
        g_mean=global_data_config['mean'],  #IMAGENET_DEFAULT_MEAN,
        g_std=global_data_config['std'],  #IMAGENET_DEFAULT_STD,
        g_crop_pct=global_data_config['crop_pct'], #None,
    )
    #assert(1==2)

    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                get_model_prefix(args),
                #safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        #saver = CheckpointSaver(
        saver = HybridCheckpointSaver(
            model=model, optimizer=optimizer, 
            global_model=global_model, global_optimizer=global_optimizer, routing_model_ema=routing_model_ema, 
            routing_model=routing_model, routing_optimizer=routing_optimizer, global_model_ema=global_model_ema,
            args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)


    if args.n_parts == 3:
        PARTS = ['gating', 'teacher', 'student']
    elif args.n_parts == 2:
        PARTS = ['gating', 'student']
    else: 
        PARTS = ['gating']

    #validate(model, global_model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
    #return
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            #eval_metrics = validate(model, global_model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            #eval_metrics = hybrid_validate(model, global_model, routing_model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            #return 
            #train_metrics = train_one_epoch(
            #    epoch, model, loader_train, optimizer, train_loss_fn, args,
            #    g_model=global_model, g_optimizer=global_optimizer, g_lr_scheduler=global_lr_scheduler, g_model_ema=global_model_ema,
            #    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
            #    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            #for part in ['gating', 'student', 'teacher']:
            for part in PARTS:
              train_metrics = hybrid_train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                r_model=routing_model, r_optimizer=routing_optimizer, r_lr_scheduler=routing_lr_scheduler, r_model_ema=routing_model_ema,
                g_model=global_model, g_optimizer=global_optimizer, g_lr_scheduler=global_lr_scheduler, g_model_ema=global_model_ema,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn, part=part, part_iters=part_iters)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
                distribute_bn(global_model, args.world_size, args.dist_bn == 'reduce')
                distribute_bn(routing_model, args.world_size, args.dist_bn == 'reduce')

            #eval_metrics = validate(model, global_model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            eval_metrics = hybrid_validate(model, global_model, routing_model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            #return

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                    distribute_bn(global_model_ema, args.world_size, args.dist_bn == 'reduce')
                    distribute_bn(routing_model_ema, args.world_size, args.dist_bn == 'reduce')
                #ema_eval_metrics = validate(model_ema.module, global_model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                ema_eval_metrics = hybrid_validate(model_ema.module, global_model_ema.module, routing_model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
                global_lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
                routing_lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))



def _focalLoss(input, target, weights, gamma=0, eps=1e-7):
        y = F.one_hot(target)
        y = y.float()

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(eps, 1. - eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** gamma # focal loss

        loss = torch.sum(loss, dim=1)
        loss = torch.sum(loss * weights)

        return loss #.sum()


def get_gate_pred( gate ) :
    gate_pred = torch.argmax(gate, dim=1) 
    return gate_pred

def get_loss( g_data, data, target, model, global_model, gating_model, args, part, amp_autocast=suppress ):
    alpha = 0.2 #args.alpha
    temperature = 1.0 #4.0 #args.temperature

    with amp_autocast():
      if part=='gating':
        with torch.no_grad():
            t_logits, _ = global_model(g_data)
            s_logits, s_ft = model(data)
        gate = gating_model( s_logits, s_ft )
      elif part=='student':
        s_logits, s_ft = model(data)
        with torch.no_grad():
            t_logits, _ = global_model(g_data)
            gate = gating_model( s_logits, s_ft )
      elif part=='teacher':
        t_logits, _ = global_model(g_data)
        with torch.no_grad():
            s_logits, s_ft = model(data)
            gate = gating_model( s_logits, s_ft )
      else:
        assert(1==2) 

    t_pred = torch.argmax( t_logits, dim=1 )
    s_pred = torch.argmax( s_logits, dim=1 )
    gate_pred = get_gate_pred( gate )

    labels = s_pred == target
    if args.strategy != 1:
        labels = torch.logical_or( t_pred!=target, labels )
    labels = labels*1

    #n_pos = torch.sum(labels == 1)
    #n_neg = torch.sum(labels == 0)
    #print('\t\tn_pos=', n_pos.item(), ' -- n_neg=', n_neg.item())

    gate_acc = torch.mean( (gate_pred == labels)*1.0 )

    valid_pnts = (gate_pred == 1) *1.0
    coverage = torch.mean( valid_pnts )
    acc_at_cov = torch.sum( valid_pnts *  torch.eq( s_pred, target ) ) / torch.sum( valid_pnts )

    valid_pnts = (gate_pred == 0) *1.0
    t_acc_at_cov = torch.sum( valid_pnts *  torch.eq( t_pred, target ) ) / torch.sum( valid_pnts )

    valid_pnts = (s_pred == t_pred) *1.0
    oracle_at_cov = torch.sum( valid_pnts *  torch.eq( s_pred, target ) ) / torch.sum( valid_pnts )

    cov_oracle = torch.mean( torch.eq(t_pred, s_pred) *1.0 )


    if part=='gating':
        g = gate.detach()
        wt_pos = torch.sum(labels==1)
        wt_neg = torch.sum(labels==0)
        wt_max = torch.max( wt_pos, wt_neg )
        wt_pos = wt_max / wt_pos
        wt_neg = wt_max /(args.g_denom* wt_neg)

        y = target
        #weights = ( (labels==0)* wt_neg + (labels==1)*wt_pos )
        #weights = ( (labels==0)* args.wneg + (labels==1)* args.wpos ) #- (args.wpos - 0.2) * (s_pred!=y) * (t_pred!=y)
        #weights = ( (labels==0)*1.2 + (labels==1)*1. )

        weights = ( (labels==0)*1. + (labels==1)*2. )
        #weights = ( (labels==0)*2. + (labels==1)*1. )
        #weights = ( (labels==0)*1. + (labels==1)*3. )
        #gate_loss = F.multi_margin_loss( gate, labels, reduction='none' )  

        #gate_loss = F.cross_entropy( gate, labels, reduction='none' )  
        #clf_loss = torch.mean( weights * gate_loss )

        clf_loss = 0.5 * _focalLoss(gate, labels, weights, gamma=2.0, eps=1e-7)
        #clf_loss = torch.mean( weights * F.cross_entropy( gate, labels, reduction='none' ) )

        Wt=100. #1000.
        Vt=0.1 #1. #0.1
        cov_loss  = Wt * F.relu( torch.mean( torch.clamp(F.relu( gate[:,0]-gate[:,1] ), max=Vt) ) - Vt*args.cov) #0.45 )
        cov_loss2 = Wt * F.relu( torch.mean( torch.clamp(F.relu( gate[:,1]-gate[:,0] ), max=Vt) ) - Vt*args.cov2) #0.45 )
        #cov_loss  = 1000. * F.relu( torch.mean( torch.clamp(F.relu( gate[:,0]-gate[:,1] ), max=0.1) ) - 0.1*args.cov) #0.45 )
        #cov_loss2 = 1000. * F.relu( torch.mean( torch.clamp(F.relu( gate[:,1]-gate[:,0] ), max=0.1) ) - 0.1*args.cov2) #0.45 )
        
        #weights = torch.clamp(F.relu( gate[:,1]-gate[:,0] ), max=1.) 
        #s_loss = torch.mean( F.cross_entropy(s_logits, target, reduction='none') *  weights )

        #weights = torch.clamp(F.relu( gate[:,0]-gate[:,1] ), max=1.) 
        #t_loss = torch.mean( F.cross_entropy(t_logits, target, reduction='none') * weights ) 

        loss = clf_loss + cov_loss + cov_loss2
        #loss = clf_loss + cov_loss + s_loss + t_loss + cov_loss2
        #loss = clf_loss #+ cov_loss 
        #print('\t\t[Gating] clf=', clf_loss.item(), ' -- cov=', cov_loss.item(), ' -- cov2=', cov_loss2.item(), ' -- s_loss=', s_loss.item(), ' -- t_loss=', t_loss.item()) #, ' l1= ', l1_loss.item())
        #print('\t\t[Gating] clf=', clf_loss.item(), ' -- cov=', cov_loss.item(), ' -- cov2=', cov_loss2.item()) #, ' l1= ', l1_loss.item())

    elif part=='teacher':
        weights = (1. * (gate_pred==1) + 1.5 * (gate_pred==0))
        #weights = (1. * (labels==1) + 1.5 * (labels==0))
        loss = torch.mean( F.cross_entropy(t_logits, target, reduction='none') * weights ) 

    elif part=='student':
        #weights = (1.5 * (labels==1) + 1. * (labels==0))
        #weights = (1.2 * (labels==1) + 1. * (labels==0))
        #weights = (2. * (gate_pred==1) + 1. * (gate_pred==0))
        #weights = (5. * (gate_pred==1) + 1. * (gate_pred==0))
        weights = (10. * (gate_pred==1) + 1. * (gate_pred==0))
        #weights = (1.2 * (gate_pred==1) + 1. * (gate_pred==0))

        s_targets = target #torch.argmax(t_logits, dim=1)  # targets
        loss = torch.mean( F.cross_entropy(s_logits, s_targets, reduction='none') *  weights )
        #loss += torch.mean( F.multi_margin_loss(s_logits, s_targets, reduction='none') * (1.0 - alpha) * weights )
        clf_loss = loss

    return loss, s_logits, t_logits, oracle_at_cov, coverage, t_acc_at_cov, acc_at_cov, gate_acc
  

def hybrid_train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        g_model=None, g_optimizer=None, g_lr_scheduler=None, g_model_ema=None,
        r_model=None, r_optimizer=None, r_lr_scheduler=None, r_model_ema=None,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, part='student', part_iters=None):


    # switch to train mode
    model.eval()
    g_model.eval()
    r_model.eval()
    if part=='gating':
        r_model.train()
    elif part=='teacher':
        g_model.train()
    elif part=='student':
        model.train()

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    #model.train()
    #g_model.train()
    #r_model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, g_input, target) in enumerate(loader):
        if batch_idx>part_iters[part]: break

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, g_input, target = input.cuda(), g_input.cuda(), target.cuda()
            if mixup_fn is not None:
                assert(1==2)
                #input, target = mixup_fn(input, target)
                #g_input, g_target = mixup_fn(g_input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
            g_input = g_input.contiguous(memory_format=torch.channels_last)

        loss, s_logits, t_logits, oracle_at_cov, coverage, t_acc_at_cov, acc_at_cov, gate_acc = get_loss( g_input, input, target, model, g_model, r_model, args, part=part, amp_autocast=amp_autocast )

        output, g_output = s_logits, t_logits

        '''with amp_autocast():
            output, _ft = model(input)
            loss = loss_fn(output, target)

            g_output, g_ft = g_model(g_input)
            g_loss = loss_fn(g_output, target)'''

        if not args.distributed:
            #losses_m.update(loss.item() + g_loss.item(), input.size(0))
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        g_optimizer.zero_grad()
        r_optimizer.zero_grad()
        if loss_scaler is not None:
            if part=='student':
              loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
            if part=='teacher':
              loss_scaler(
                loss, g_optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(g_model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
            if part=='gating':
              loss_scaler(
                loss, r_optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(r_model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                if part=='student':
                  dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
                if part=='teacher':
                  dispatch_clip_grad(
                    model_parameters(g_model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
                if part=='gating':
                  dispatch_clip_grad(
                    model_parameters(r_model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            if part=='student':
              optimizer.step()
            if part=='teacher':
              g_optimizer.step()
            if part=='gating':
              r_optimizer.step()

        if model_ema is not None:
            if part=='student':
              model_ema.update(model)
            if part=='teacher':
              g_model_ema.update(g_model)
            if part=='gating':
              r_model_ema.update(r_model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            r_lrl = [param_group['lr'] for param_group in r_optimizer.param_groups]
            r_lr = sum(r_lrl) / len(r_lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                #g_reduced_loss = reduce_tensor(g_loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  r-LR: {r_lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        r_lr=r_lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            if part=='student':
              lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
            if part=='teacher':
              g_lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
            if part=='gating':
              r_lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])





def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        g_model=None, g_optimizer=None, g_lr_scheduler=None, g_model_ema=None,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    g_model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, g_input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, g_input, target = input.cuda(), g_input.cuda(), target.cuda()
            if mixup_fn is not None:
                assert(1==2)
                #input, target = mixup_fn(input, target)
                #g_input, g_target = mixup_fn(g_input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
            g_input = g_input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output, _ft = model(input)
            loss = loss_fn(output, target)

            g_output, g_ft = g_model(g_input)
            g_loss = loss_fn(g_output, target)

        if not args.distributed:
            losses_m.update(loss.item() + g_loss.item(), input.size(0))

        optimizer.zero_grad()
        g_optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
            loss_scaler(
                g_loss, g_optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(g_model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            g_loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
                dispatch_clip_grad(
                    model_parameters(g_model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()
            g_optimizer.step()

        if model_ema is not None:
            model_ema.update(model)
            g_model_ema.update(g_model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                g_reduced_loss = reduce_tensor(g_loss.data, args.world_size)
                losses_m.update(reduced_loss.item() + g_reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
            g_lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])



def hybrid_validate(model, g_model, routingNet, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    Tlosses_m = AverageMeter()
    Ttop1_m = AverageMeter()
    Ttop5_m = AverageMeter()

    h_losses_m = AverageMeter()
    oracleAtCov = AverageMeter()
    tAtCov = AverageMeter()
    sAtCov = AverageMeter()
    lcov = AverageMeter()
    gateAcc = AverageMeter()

    model.eval()
    g_model.eval()
    routingNet.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, g_input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                g_input = g_input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
                g_input = g_input.contiguous(memory_format=torch.channels_last)

            '''with amp_autocast():
                s_logits, s_ft = model(input)
                t_logits, _ = g_model(g_input)
                gate = routingNet( s_logits, s_ft )'''

            h_loss, s_logits, t_logits, oracle_at_cov, coverage, t_acc_at_cov, acc_at_cov, gate_acc = get_loss( g_input, input, target, model, g_model, routingNet, args, part='student', amp_autocast=amp_autocast )

            output, g_output = s_logits, t_logits

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            Tloss = loss_fn(g_output, target)
            Tacc1, Tacc5 = accuracy(g_output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)

                Treduced_loss = reduce_tensor(Tloss.data, args.world_size)
                Tacc1 = reduce_tensor(Tacc1, args.world_size)
                Tacc5 = reduce_tensor(Tacc5, args.world_size)

                h_loss = reduce_tensor(h_loss, args.world_size)
                gate_acc = reduce_tensor(gate_acc, args.world_size)
                coverage = reduce_tensor(coverage, args.world_size)
                t_acc_at_cov = reduce_tensor(t_acc_at_cov, args.world_size)
                acc_at_cov = reduce_tensor(acc_at_cov, args.world_size)
                oracle_at_cov = reduce_tensor(oracle_at_cov, args.world_size)
            else:
                reduced_loss = loss.data
                Treduced_loss = Tloss.data

            torch.cuda.synchronize()

            h_losses_m.update(h_loss.item(), input.size(0))

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            Tlosses_m.update(Treduced_loss.item(), input.size(0))
            Ttop1_m.update(Tacc1.item(), g_output.size(0))
            Ttop5_m.update(Tacc5.item(), g_output.size(0))

            oracleAtCov.update(oracle_at_cov.item(), input.size(0) )
            if (math.isnan(t_acc_at_cov.item())==False) and (math.isnan(acc_at_cov.item())==False) : 
                lcov.update(coverage.item(), input.size(0) )
                tAtCov.update(t_acc_at_cov.item(), input.size(0) )
                sAtCov.update(acc_at_cov.item(), input.size(0) )
            gateAcc.update( gate_acc.item(), input.size(0) )


            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):

                hybrid_flops = args.base_flops + (1. - lcov.avg) * args.global_flops
                hybrid_acc = ( lcov.avg * sAtCov.avg + (1. - lcov.avg) * tAtCov.avg ) * 100
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: ({batch_time.avg:.3f}) '
                    'HLoss: ({h_loss.avg:>6.4f}) '
                    'Acc@1: ({top1.avg:>7.4f}) '
                    'TAcc@1: ({Ttop1.avg:>7.4f}) '
                    'HFlops: ({hybrid_flops:>7.4f}) '
                    'HAcc: ({hybrid_acc:>7.4f}) '
                    'Cov: ({lcov.avg:>7.4f}) '
                    'OAtCov: ({oracleAtCov.avg:>7.4f}) '
                    'tAtCov: ({tAtCov.avg:>7.4f}) '
                    'sAtCov: ({sAtCov.avg:>7.4f}) '
                    'RAcc@1: ({gateAcc.avg:>7.4f}) '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m, h_loss=h_losses_m,
                        hybrid_flops=hybrid_flops, hybrid_acc=hybrid_acc,
                        lcov=lcov, tAtCov=tAtCov, sAtCov=sAtCov, oracleAtCov=oracleAtCov, gateAcc=gateAcc,
                        Tloss=Tlosses_m, Ttop1=Ttop1_m, Ttop5=Ttop5_m,))

    hybrid_flops = args.base_flops + (1. - lcov.avg) * args.global_flops
    hybrid_acc = ( lcov.avg * sAtCov.avg + (1. - lcov.avg) * tAtCov.avg ) * 100
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg),
                           ('Tloss', Tlosses_m.avg), ('Ttop1', Ttop1_m.avg), ('Ttop5', Ttop5_m.avg),
                           ('cov', lcov.avg), ('gateAcc', gateAcc.avg), ('oracleAtCov', oracleAtCov.avg), 
                           ('tAtCov', tAtCov.avg), ('sAtCov', sAtCov.avg), ('h_loss', h_losses_m.avg), 
                           ('hybrid_flops', hybrid_flops),  ('hybrid_acc', hybrid_acc), 
                          ])

    return metrics



def validate(model, g_model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    Tlosses_m = AverageMeter()
    Ttop1_m = AverageMeter()
    Ttop5_m = AverageMeter()
    model.eval()
    g_model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, g_input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                g_input = g_input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
                g_input = g_input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output, _ft = model(input)
                g_output, _ft = g_model(g_input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            if isinstance(g_output, (tuple, list)):
                g_output = g_output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            Tloss = loss_fn(g_output, target)
            Tacc1, Tacc5 = accuracy(g_output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)

                Treduced_loss = reduce_tensor(Tloss.data, args.world_size)
                Tacc1 = reduce_tensor(Tacc1, args.world_size)
                Tacc5 = reduce_tensor(Tacc5, args.world_size)
            else:
                reduced_loss = loss.data
                Treduced_loss = Tloss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            Tlosses_m.update(Treduced_loss.item(), input.size(0))
            Ttop1_m.update(Tacc1.item(), g_output.size(0))
            Ttop5_m.update(Tacc5.item(), g_output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                    'TLoss: {Tloss.val:>7.4f} ({Tloss.avg:>6.4f}) '
                    'TAcc@1: {Ttop1.val:>7.4f} ({Ttop1.avg:>7.4f}) '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m,
                        Tloss=Tlosses_m, Ttop1=Ttop1_m, Ttop5=Ttop5_m,))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg),
                           ('Tloss', Tlosses_m.avg), ('Ttop1', Ttop1_m.avg), ('Ttop5', Ttop5_m.avg),
                          ])

    return metrics


if __name__ == '__main__':
    main()
