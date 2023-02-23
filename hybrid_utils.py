
import sys
sys.path.append(".")

import time
from copy import deepcopy
import shutil
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.distributions import Categorical
from torchvision import datasets, transforms
import os
import logging
import math
from tqdm import tqdm
import json
from utils import DistributedMetric, accuracy #, AverageMeter 
from mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small

from tinynas.nn.networks import ProxylessNASNets
from torch.utils.data import Dataset
from model_ema import ModelEmaV2

from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.model_zoo import ofa_net, ofa_specialized

import timm 
from custom_transforms import autoaugment
from utils import count_net_flops
from cosine_lr import CosineLRScheduler

from routing_models import RoutingNetworkWithLogits, RoutingNetworkWithFt, RoutingNetworkWithExtraFt, RoutingNetwork, RoutingNetworkTop20, RoutingNetworkTop20WithFt

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

def cnt_free_params(layer):
    cnt = 0
    for param in layer.parameters():
        if param.requires_grad: cnt += 1
    return cnt

def freeze_layer_params(layer):
    for param in layer.parameters():
        param.requires_grad = False

class TimmModel(nn.Module):
    def __init__(self, model, model_name):
        super(TimmModel, self).__init__()

        self.is_efficientnet = 'efficient' in model_name
        self.model = model
        if self.is_efficientnet:
            self.global_pool = model.global_pool
        else:
            self.flatten = model.flatten
        self.classifier = model.classifier
        self.drop_rate = model.drop_rate
        self.skip_len = 2

    def freeze_bn(self):
        #self.model.conv_stem.eval()
        #self.model.blocks.eval()
        #self.model.conv_head.eval()
        for name, layer in self.model.named_children():
            if 'classifier' not in name:
                layer.eval()

    def freeze_backbone(self):
        '''for name, layer in self.model.named_children():
            if 'classifier' not in name:
                #if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    print('layer -- ', name)
                layer.eval()
                freeze_layer_params(layer)'''
        #assert(1==2)
        freeze_layer_params(self.model.conv_stem)

        print(len(self.model.blocks))
        for i in range(len(self.model.blocks)-self.skip_len):
            layer = self.model.blocks[i]
            freeze_layer_params(layer)


    def forward(self, x):
        x = self.model.forward_features(x)
        
        if self.is_efficientnet:
            x = self.global_pool(x)
        else:
            x = self.flatten(x)

        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        ft = x
        x = self.classifier(x)
        return x, ft


class OFANetModel(nn.Module):
    def __init__(self, model):
        super(OFANetModel, self).__init__()

        self.model = model
        self.first_conv = model.first_conv
        self.blocks = model.blocks
        self.final_expand_layer = model.final_expand_layer
        self.global_avg_pool = model.global_avg_pool
        self.feature_mix_layer = model.feature_mix_layer
        self.classifier = model.classifier
        self.skip_len = 2

    def freeze_bn(self):
        print('OFA.. not freezing batch norm stats.. ')
        '''print('OFA.. freezing batch norm stats.. ')
        self.first_conv.eval()
        self.blocks.eval() 
        self.final_expand_layer.eval()
        self.feature_mix_layer.eval()'''

    def freeze_backbone(self):
        print('OFA.. not freezing backbone.. ')
        '''self.first_conv.eval()
        self.blocks.eval() 
        self.final_expand_layer.eval()
        self.feature_mix_layer.eval()

        #for layer in [ self.first_conv, self.blocks, 
        #        self.final_expand_layer, self.feature_mix_layer ]:
        #    freeze_layer_params(layer)

        freeze_layer_params(self.first_conv)

        print(len(self.blocks))
        for i in range(len(self.blocks)-self.skip_len):
            layer = self.blocks[i]
            freeze_layer_params(layer)'''


    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = self.global_avg_pool(x)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        ft = x
        x = self.classifier(x)
        return x, ft


class Ft_OFANetModel(nn.Module):
    def __init__(self, model):
        super(Ft_OFANetModel, self).__init__()

        self.model = model
        self.first_conv = model.first_conv
        self.blocks = model.blocks
        self.final_expand_layer = model.final_expand_layer
        self.global_avg_pool = model.global_avg_pool
        self.feature_mix_layer = model.feature_mix_layer
        self.classifier = model.classifier
        self.skip_len = 2

    def freeze_bn(self):
        print('OFA.. not freezing batch norm stats.. ')
        '''print('OFA.. freezing batch norm stats.. ')
        self.first_conv.eval()
        self.blocks.eval() 
        self.final_expand_layer.eval()
        self.feature_mix_layer.eval()'''

    def freeze_backbone(self):
        print('OFA.. not freezing backbone.. ')
        '''self.first_conv.eval()
        self.blocks.eval() 
        self.final_expand_layer.eval()
        self.feature_mix_layer.eval()

        #for layer in [ self.first_conv, self.blocks, 
        #        self.final_expand_layer, self.feature_mix_layer ]:
        #    freeze_layer_params(layer)

        freeze_layer_params(self.first_conv)

        print(len(self.blocks))
        for i in range(len(self.blocks)-self.skip_len):
            layer = self.blocks[i]
            freeze_layer_params(layer)'''


    def forward(self, x):
        ft = [ x  ]
        x = self.first_conv(x)
        ft.append(x)
        for block in self.blocks:
            x = block(x)
            ft.append(x)
        x = self.final_expand_layer(x)
        ft.append(x)
        x = self.global_avg_pool(x)  # global average pooling
        ft.append(x)
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        ft.append(x)
        #ft = x
        x = self.classifier(x)
        return x, ft



class MCUNetModel(nn.Module):
    def __init__(self, model):
        super(MCUNetModel, self).__init__()

        self.model = model
        self.first_conv = model.first_conv
        self.blocks = model.blocks
        self.feature_mix_layer = model.feature_mix_layer
        self.classifier = model.classifier
        self.skip_len = 2

    def freeze_bn(self):
        print('MCUNet.. freezing batch norm stats.. ')
        self.first_conv.eval()
        self.blocks.eval() 
        if self.feature_mix_layer is not None:
            self.feature_mix_layer.eval()

    def freeze_backbone(self):
        self.first_conv.eval()
        self.blocks.eval() 
        if self.feature_mix_layer is not None:
            self.feature_mix_layer.eval()
            #freeze_layer_params(self.feature_mix_layer)

        freeze_layer_params(self.first_conv)

        print(len(self.blocks))
        for i in range(len(self.blocks)-self.skip_len):
            layer = self.blocks[i]
            freeze_layer_params(layer)


    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        ft = x
        x = self.classifier(x)
        return x, ft

def get_entropy_thr_at_cov(y_entropy, target_cov=0.9, num=50, low=0, high=5):
    best_thr=-10.0 #-1.84
    _best_cov = 0.99
    for thr in list(np.linspace(low, high, num=num)):
        _cov = np.mean( (y_entropy <= thr)*1 )
        if _cov >= target_cov and _cov<=_best_cov: 
            _best_cov = _cov
            best_thr = thr
    return best_thr, _best_cov


def get_thr_at_cov(y_gate_vals, target_cov=0.9, num=50, low=-2, high=2):
    best_thr=-10.0 #-1.84
    _best_cov = 0.99
    for thr in list(np.linspace(low, high, num=num)):
        _cov = np.mean( ((y_gate_vals[:,1]-y_gate_vals[:,0]) >= thr)*1  )
        if _cov >= target_cov and _cov<=_best_cov: 
            _best_cov = _cov
            best_thr = thr
    return best_thr, _best_cov


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', default=None, type=str)
    parser.add_argument('--log-dir', default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoint',
                        help='checkpoint file format')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    # architecture setting
    parser.add_argument('--cov', type=float, default=0.55, help='target 1-coverage term in the formulation.')
    parser.add_argument('--g_denom', type=float, default=1., help='denominator in the balanced routing loss formulation.')
    parser.add_argument('--strategy', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--kt', type=int, default=1, help='number of epochs per alternates')
    parser.add_argument('--base_bootstrap', type=int, default=0, help='batches for base optimization')
    parser.add_argument('--base_use_scheduler', type=int, default=1, help='batches for base optimization')
    parser.add_argument('--base_backbone_freeze', type=int, default=1, help='batches for base optimization')
    parser.add_argument('--base_bn_freeze', type=int, default=1, help='batches for base optimization')
    parser.add_argument('--base_opt_type', default='adam', type=str)
    parser.add_argument('--base_type', default='mcunet', type=str)
    parser.add_argument('--base_arch', default='mcunet-5fps_imagenet', type=str)
    parser.add_argument('--s_warmup_lr', type=float, default=0.000001, help='learning rate for base')
    parser.add_argument('--s_lr', type=float, default=0.0001, help='learning rate for base')
    parser.add_argument('--s_iters', type=int, default=5005, help='batches for base optimization')

    parser.add_argument('--warmup_t', type=int, default=3, help='number of epochs to train')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='learning rate for base')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='learning rate for base')

    parser.add_argument('--global_bootstrap', type=int, default=0, help='batches for base optimization')
    parser.add_argument('--global_use_scheduler', type=int, default=1, help='batches for base optimization')
    parser.add_argument('--global_backbone_freeze', type=int, default=1, help='batches for base optimization')
    parser.add_argument('--global_bn_freeze', type=int, default=1, help='batches for base optimization')
    parser.add_argument('--global_opt_type', default='adam', type=str)
    parser.add_argument('--global_type', default='ofa', type=str)
    parser.add_argument('--global_arch', default='mcunet_base_fixed_global_search', type=str)
    parser.add_argument('--t_warmup_lr', type=float, default=0.000001, help='learning rate for base')
    parser.add_argument('--t_lr', type=float, default=0.0001, help='learning rate for global')
    parser.add_argument('--t_iters', type=int, default=5005, help='batches for global optimization')

    parser.add_argument('--routing_use_scheduler', type=int, default=1, help='batches for base optimization')
    parser.add_argument('--routing_opt_type', default='adam', type=str)
    parser.add_argument('--routing_arch', default='with_ft', type=str)
    parser.add_argument('--g_warmup_lr', type=float, default=0.000001, help='learning rate for base')
    parser.add_argument('--g_lr', type=float, default=0.0004, help='learning rate for gate')
    parser.add_argument('--g_iters', type=int, default=505, help='batches for gate optimization')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='proxyless')
    parser.add_argument('--net_config', default=None, type=str)
    # data setting
    parser.add_argument('--path', help='The path of imagenet',  type=str, default='/mnt/active/datasets/imagenet')
    parser.add_argument('--train-dir', default=os.path.expanduser('/dataset/imagenet/train'),
                        help='path to training data')
    parser.add_argument('--val-dir', default=os.path.expanduser('/dataset/imagenet/val'),
                        help='path to validation data')
    parser.add_argument('--resolution', default=None, type=int)  # will set from model config
    # training hyper-params
    parser.add_argument('--n_parts', type=int, default=1,
                        help='input batch size for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00001,  help='weight decay')
    parser.add_argument('--r_wd', type=float, default=0.001,  help='weight decay')
    parser.add_argument('--lr_type', type=str, default='cosine')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers')
    # resuming from previous weights
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--load_from', default=None, type=str,
                        help='load from a checkpoint, either for evaluation or fine-tuning')
    
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--calibrate', action='store_true', default=False)
    # extra techniques (not used for paper results)
    parser.add_argument('--mixup-alpha', default=0, type=float, help='The alpha value used in mix up training')
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--random-erase-prob', default=0.2, type=float, help='random erasing probability (default: 0.0)')
    return parser
    
class TwoDatasets(Dataset):
    def __init__(self, ds1, ds2):
       self.ds1 = ds1
       self.ds2 = ds2
       assert( len(self.ds1) == len(self.ds2) )

    def __len__(self):
       return len(self.ds1) 

    def __getitem__(self, idx):
       image1, label1 = self.ds1[idx]
       image2, label2 = self.ds2[idx]
       return (image1, label1, image2, label2)

def get_base_global_loaders( train_dir, val_dir, workers=16, 
            global_model_cfg=None, base_model_cfg=None, 
            base_pct=0.875, global_pct=0.875,
            base_resolution = 224, global_resolution = 224, 
            base_normalize = None, global_normalize = None,
            random_erase_prob = 0.2, batch_size=1024, eval_batch_size=1024, prefetch_factor=2, 
              ):
    kwargs = {'num_workers': workers, 'pin_memory': True, 'prefetch_factor' : prefetch_factor}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if base_normalize is None:
        base_normalize = normalize
    if global_normalize is None:
        global_normalize = normalize

    if base_model_cfg is not None:
        input_size = base_model_cfg['input_size']
        interpolation = base_model_cfg['interpolation']
        base_resolution = input_size[2] 
        base_normalize = transforms.Normalize( mean=base_model_cfg['mean'], std=base_model_cfg['std'] )
        base_pct = base_model_cfg['crop_pct']
  
    if global_model_cfg is not None:
        input_size = global_model_cfg['input_size']
        interpolation = global_model_cfg['interpolation']
        global_resolution = input_size[2] 
        global_normalize = transforms.Normalize( mean=global_model_cfg['mean'], std=global_model_cfg['std'] )
        global_pct = global_model_cfg['crop_pct']

    train_dataset = datasets.ImageFolder(train_dir,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(base_resolution),
                                         transforms.RandomHorizontalFlip(),
                                         #autoaugment.AutoAugment( autoaugment.AutoAugmentPolicy('imagenet') ),
                                         #transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                                         transforms.ToTensor(),
                                         base_normalize,
                                         #transforms.RandomErasing(p=random_erase_prob)
                                     ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,**kwargs)

    val_dataset = datasets.ImageFolder(val_dir,
                                   transform=transforms.Compose([
                                       transforms.Resize(int(base_resolution / base_pct)),
                                       transforms.CenterCrop(base_resolution),
                                       transforms.ToTensor(),
                                       base_normalize
                                   ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size, **kwargs) #sampler=val_sampler, **kwargs)


    global_train_dataset = datasets.ImageFolder(train_dir,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(global_resolution),
                                         transforms.RandomHorizontalFlip(),
                                         #autoaugment.AutoAugment( autoaugment.AutoAugmentPolicy('imagenet') ),
                                         #transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                                         transforms.ToTensor(),
                                         global_normalize,
                                         #transforms.RandomErasing(p=random_erase_prob)
                                     ]))
    global_train_loader = torch.utils.data.DataLoader(global_train_dataset, batch_size=batch_size,**kwargs)

    global_val_dataset = datasets.ImageFolder(val_dir,
                                   transform=transforms.Compose([
                                       transforms.Resize(int(global_resolution / global_pct)),
                                       transforms.CenterCrop(global_resolution),
                                       transforms.ToTensor(),
                                       global_normalize
                                   ]))
    global_val_loader = torch.utils.data.DataLoader(global_val_dataset, batch_size=eval_batch_size, **kwargs) #sampler=val_sampler, **kwargs)

    joint_train_dataset = TwoDatasets( train_dataset, global_train_dataset )
    joint_val_dataset = TwoDatasets( val_dataset, global_val_dataset )
    #joint_train_loader = torch.utils.data.DataLoader(joint_train_dataset, batch_size=batch_size,**kwargs)
    joint_train_loader =  MultiEpochsDataLoader(joint_train_dataset, batch_size=batch_size,**kwargs)
    joint_val_loader = torch.utils.data.DataLoader(joint_val_dataset, batch_size=eval_batch_size, **kwargs) #sampler=val_sampler, **kwargs)

    return joint_train_loader, joint_val_loader, train_loader, val_loader, global_train_loader, global_val_loader


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def freeze_layer_name(name):
    if name.startswith( str(0) + '.' ): return True
    for i in range(8):
        #if str(i) + '.block.' in name: return True
        if name.startswith( str(i) + '.block.' ): return True
    return False

def set_bn_features_eval(model):
  names_list = [ str(i) for i in range(8) ]
  for name, m in model.features.named_children():
      if name in names_list:
          m.eval()
      else:
        print(name)

def freeze_features(model):
  for name, param in model.features.named_parameters():
    if freeze_layer_name(name):
        param.requires_grad = False
    else:
        print(name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='Default', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




def get_hybrid_stats( base_y_pred, global_y_pred, base_y_true, gate_base, gate_global, 
           base_model_stats, global_model_stats, hybrid_model_stats, prefix='entropy',
           base_name='', global_name='' ):
    hybrid_pred = ( base_y_pred * gate_base ) + ( global_y_pred * gate_global )  
    hybrid_cov = np.sum( gate_base ) / len(base_y_true)
    hybrid_acc = np.sum( hybrid_pred == base_y_true ) / len(base_y_true)
    hybrid_flops = base_model_stats[base_name+'flop'] + (1. - hybrid_cov) * global_model_stats[global_name+'flop']

    global_abstained_acc = np.sum( (global_y_pred == base_y_true) * gate_global ) / np.sum( gate_global )
    base_predicted_acc = np.sum( (base_y_pred == base_y_true) * gate_base ) / np.sum( gate_base )

    hybrid_model_stats[prefix + '_hybrid-valid_acc'] = hybrid_acc*100
    hybrid_model_stats[prefix + '_hybrid-cov'] = hybrid_cov*100
    hybrid_model_stats[prefix + '_hybrid-flop'] = hybrid_flops
    hybrid_model_stats[prefix + '_hybrid-global_abs_acc'] = global_abstained_acc*100
    hybrid_model_stats[prefix + '_hybrid-base_pred_acc'] = base_predicted_acc*100

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

'''
class RoutingNetworkWithLogits(nn.Module):
    def __init__(self, num_ft):
        super(RoutingNetworkWithLogits, self).__init__()
        self.flatten = nn.Flatten()
        n_ft = 1 + 1000 #1000 + 1 + 100 + 576 #1024
        n_ft2 = 64
        print('Total routing ft = ', n_ft)
        self.routing = nn.Sequential(
            nn.Linear( n_ft, n_ft2, bias=True),
            nn.BatchNorm1d( n_ft2 ),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True),
        )
        self.routing.apply(init_weights)

    def forward(self, logits, ft):
        entropy = Categorical( logits=logits ).entropy().unsqueeze(1)

        x = torch.cat([entropy, logits], dim=1)

        x = self.flatten(x)
        gate = self.routing(x)
        return gate



class RoutingNetworkWithFt(nn.Module):
    def __init__(self, num_ft):
        super(RoutingNetworkWithFt, self).__init__()
        self.flatten = nn.Flatten()
        n_ft = 101 + num_ft + 1000 + 1000 #1000 + 1 + 100 + 576 #1024
        n_ft2 = 64
        print('Total routing ft = ', n_ft)
        self.routing = nn.Sequential(
            nn.BatchNorm1d( n_ft ),
            nn.Linear( n_ft, n_ft2, bias=True),
            nn.BatchNorm1d( n_ft2 ),
            nn.ReLU(),
            nn.Linear(n_ft2, 64, bias=True),
            nn.BatchNorm1d( 64 ),
            nn.Linear(64, 2, bias=True),
        )
        self.routing.apply(init_weights)

    def forward(self, logits, ft):
        softmax = F.softmax( logits, dim=1 )
        entropy = Categorical( probs=softmax ).entropy().unsqueeze(1)
        #logs = torch.log( softmax + 1e-7 )

        B = logits.shape[0]
        C = 10
        topk = torch.topk( softmax, C, dim=1 )[0]
        y_margin = topk.view(B, C, -1) - topk.view(B, -1, C)
        y_margin = y_margin.view(B, -1)

        #x = torch.cat([entropy, y_margin], dim=1)
        #x = torch.cat([entropy, y_margin, ft], dim=1)
        x = torch.cat([entropy, y_margin, ft, softmax, logits], dim=1)
        #x = torch.cat([entropy, softmax, y_margin, ft], dim=1)
        #x = torch.cat([entropy, logits, softmax, logs], dim=1)
        #print('x = ', x.size())

        x = self.flatten(x)
        gate = self.routing(x)
        #print('gate = ', gate.size())
        #assert(1==2)
        #return ft, logits, gate
        return gate

class RoutingNetworkWithExtraFt(nn.Module):
    def __init__(self, num_ft):
        super(RoutingNetworkWithExtraFt, self).__init__()
        self.flatten = nn.Flatten()
        self.K = 5
        n_ft = self.K * self.K + 1 + num_ft + 1000 #1000 + 1 + 100 + 576 #1024
        n_ft2 = 64
        print('ExtraFt -- Total routing ft = ', n_ft)

        n_ft = self.K *  self.K  + 1
        self.R1 = nn.Sequential(
            nn.BatchNorm1d( n_ft ),
            nn.Linear( n_ft, n_ft2, bias=True),
            nn.BatchNorm1d( n_ft2 ),
            nn.ReLU(),
        )

        n_ft = num_ft
        self.R2 = nn.Sequential(
            nn.BatchNorm1d( n_ft ),
            nn.Linear( n_ft, n_ft2, bias=True),
            nn.BatchNorm1d( n_ft2 ),
            nn.ReLU(),
        )

        n_ft = 1000
        self.R3 = nn.Sequential(
            nn.BatchNorm1d( n_ft ),
            nn.Linear( n_ft, n_ft2, bias=True),
            nn.BatchNorm1d( n_ft2 ),
            nn.ReLU(),
        )

        self.clf = nn.Sequential(
            #nn.Linear( 3 * n_ft2, 64, bias=True),
            #nn.Linear( n_ft2, 64, bias=True),
            nn.Linear( 2*n_ft2, 64, bias=True),
            nn.BatchNorm1d( 64 ),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True),
        )

        for layer in [self.R1, self.R2, self.R3, self.clf]:
            #self.routing.apply(init_weights)
            layer.apply(init_weights)

    def forward(self, logits, ft):
        softmax = F.softmax( logits, dim=1 )
        entropy = Categorical( probs=softmax ).entropy().unsqueeze(1)
        #logs = torch.log( softmax + 1e-7 )

        B = logits.shape[0]
        C = self.K
        topk = torch.topk( softmax, C, dim=1 )[0]
        y_margin = topk.view(B, C, -1) - topk.view(B, -1, C)
        y_margin = y_margin.view(B, -1)

        #x = torch.cat([entropy, y_margin], dim=1)
        #x = torch.cat([entropy, y_margin, ft], dim=1)
        x = torch.cat([entropy, y_margin], dim=1)
        #x = torch.cat([entropy, softmax, y_margin, ft], dim=1)
        #x = torch.cat([entropy, logits, softmax, logs], dim=1)
        #print('x = ', x.size())

        x = self.flatten(x)
        x1 = self.R1( torch.cat([ entropy, y_margin ], dim=1) )
        #x2 = self.R2( ft )
        #x3 = self.R3( softmax )
        x3 = self.R3( logits )

        #gate = self.clf( torch.cat([x1, x2, x3], dim=1) )
        gate = self.clf( torch.cat([x1, x3], dim=1) )
        #gate = self.clf(x1)
        #gate = self.routing(x)
        #print('gate = ', gate.size())
        #assert(1==2)
        #return ft, logits, gate
        return gate



class RoutingNetwork(nn.Module):
    def __init__(self):
        super(RoutingNetwork, self).__init__()
        self.flatten = nn.Flatten()
        n_ft = 101 #1000 + 1 + 100 + 576 #1024
        print('Top10x10 -- Total routing ft = ', n_ft)
        self.routing = nn.Sequential(
            nn.BatchNorm1d( n_ft ),
            nn.Linear( n_ft, 256, bias=True),
            nn.BatchNorm1d( 256 ),
            nn.ReLU(),
            nn.Linear(256, 64, bias=True),
            nn.BatchNorm1d( 64 ),
            nn.Linear(64, 2, bias=True),
        )
        self.routing.apply(init_weights)

    def forward(self, logits, ft):
        softmax = F.softmax( logits, dim=1 )
        entropy = Categorical( probs=softmax ).entropy().unsqueeze(1)
        #logs = torch.log( softmax + 1e-7 )

        B = logits.shape[0]
        C = 10
        topk = torch.topk( softmax, C, dim=1 )[0]
        y_margin = topk.view(B, C, -1) - topk.view(B, -1, C)
        y_margin = y_margin.view(B, -1)

        x = torch.cat([entropy, y_margin], dim=1)
        #x = torch.cat([entropy, softmax, y_margin, ft], dim=1)
        #x = torch.cat([entropy, logits, softmax, logs], dim=1)
        #print('x = ', x.size())

        x = self.flatten(x)
        gate = self.routing(x)
        #print('gate = ', gate.size())
        #assert(1==2)
        #return ft, logits, gate
        return gate
'''

def freeze_base_features(model):
    for param in model.features.parameters():
        param.requires_grad = False



def old_validate(epoch, model, loader, base=True, args=None, device='cuda'):
    model.eval()
    val_loss = AverageMeter() #DistributedMetric('val_loss')
    val_top1 = AverageMeter() #DistributedMetric('val_top1')
    val_top5 = AverageMeter() #DistributedMetric('val_top5')

    B = len(loader.dataset) // args.batch_size
    N = len(loader.dataset) #len(loader) * args.batch_size
    y_loss = np.zeros( (N,) )
    y_pred = np.zeros( (N,) )
    y_true = np.zeros( (N,) )
    y_entropy = np.zeros( (N,) )

    i = 0
    #with tqdm(total=len(loader),
    #          desc='Validate Epoch  #{}'.format(epoch + 1),
    #          disable=not verbose) as t:
    with torch.no_grad():
            for data_target in loader:
                data, target = data_target
                data, target = data.to(device), target.to(device)

                if base:
                    output, _ = model(data)
                else:
                    output = model(data)

                logits = output
                _start = i*args.batch_size
                _end   = (i+1)*args.batch_size

                if _end > N: _end=N

                pred = torch.argmax( logits, dim=1 ) # logits
                softmax = F.softmax( logits, dim=1 )
                entropy = Categorical( probs=softmax ).entropy()
                s_loss = F.multi_margin_loss( logits, target, reduction='none' )

                y_loss[ _start:_end ] = s_loss.detach().cpu().numpy()

                y_entropy[ _start:_end ] = entropy.detach().cpu().numpy()
                y_pred[ _start:_end ] = pred.detach().cpu().numpy()
                y_true[ _start:_end ] = target.detach().cpu().numpy()

                i += 1

                val_loss.update(F.cross_entropy(output, target))
                top1, top5 = accuracy(output, target, topk=(1, 5))
                val_top1.update(top1)
                val_top5.update(top5)
                #t.set_postfix({'loss': val_loss.avg.item(),
                #               'top1': val_top1.avg.item(),
                #               'top5': val_top5.avg.item()})
                #t.update(1)
                #if i>20: break
    
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=val_top1, top5=val_top5))
    return val_loss.avg, val_top1.avg, val_top5.avg, y_pred, y_true, y_entropy, y_loss

def evaluate_routing_model( routingNet, network, xloader, batch_size, base=False ):
    losses, top1, top5 = ( AverageMeter(), AverageMeter(), AverageMeter(),)

    B = len(xloader.dataset) // batch_size
    N = len(xloader.dataset) #len(loader) * args.batch_size
    y_loss = np.zeros( (N,) )
    y_pred = np.zeros( (N,) )
    y_true = np.zeros( (N,) )
    y_entropy = np.zeros( (N,) )
    y_gate = np.zeros( (N,) )
    y_gate_vals = np.zeros( (N,2) )

    network.eval()
    routingNet.eval()
    with torch.no_grad():
      for i, (inputs, targets) in enumerate(xloader):
        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)

        if base:
            logits, ft = network(inputs)
        else:
            logits = network(inputs)
            ft = None
        gates = routingNet(logits, ft)

        _start = i*batch_size
        _end   = (i+1)*batch_size
        if _end > N: _end=N

        pred = torch.argmax( logits, dim=1 ) # logits
        softmax = F.softmax( logits, dim=1 )
        entropy = Categorical( probs=softmax ).entropy()
        s_loss = F.multi_margin_loss( logits, targets, reduction='none' )

        y_loss[ _start:_end ] = s_loss.detach().cpu().numpy()
        y_entropy[ _start:_end ] = entropy.detach().cpu().numpy()
        y_pred[ _start:_end ] = pred.detach().cpu().numpy()
        y_true[ _start:_end ] = targets.detach().cpu().numpy()
        y_gate[ _start:_end ] = (torch.argmax(gates, dim=1)).detach().cpu().numpy()
        y_gate_vals[ _start:_end ] = gates.detach().cpu().numpy()

        loss = F.cross_entropy(logits, targets)

        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))

        losses.update(loss) #.item(), inputs.size(0))
        top1.update(prec1) #.item(), inputs.size(0))
        top5.update(prec5) #.item(), inputs.size(0))

        
        #torch.cuda.empty_cache()
 
    return losses.avg, top1.avg, top5.avg, y_pred, y_true, y_entropy, y_gate, y_loss, y_gate_vals

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+'model_best.pth.tar')

def get_gate_pred( gate ) :
    gate_pred = torch.argmax(gate, dim=1) 
    return gate_pred

def get_loss( g_data, data, target, model, global_model, gating_model, criterion, args, part ):
    alpha = 0.2 #args.alpha
    temperature = 1.0 #4.0 #args.temperature

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

    with torch.no_grad():
        s_loss = F.cross_entropy( s_logits, target, reduction='none' )
        t_loss = F.cross_entropy( t_logits, target, reduction='none' )

    #print('s_loss = ', torch.mean(s_loss).item())
    #print('t_loss = ', torch.mean(t_loss).item())
    #assert(1==2)

    t_pred = torch.argmax( t_logits, dim=1 )
    s_pred = torch.argmax( s_logits, dim=1 )
    gate_pred = get_gate_pred( gate )

    labels = s_pred == target
    #_lmbda=0.2
    if args.strategy != 1:
        labels = torch.logical_or( t_pred!=target, labels )
    #labels = torch.logical_or( ((s_loss - t_loss) <= _lmbda), s_pred==t_pred )
    #labels = s_pred == t_pred #torch.logical_or( ((s_loss - t_loss) <= _lmbda), s_pred==t_pred )
    labels = labels*1

    #n_pos = torch.sum(labels == 1)
    #n_neg = torch.sum(labels == 0)
    #print('\t\tn_pos=', n_pos.item(), ' -- n_neg=', n_neg.item())

    #sCorrectButGlobal = torch.sum( (s_pred == target) * (gate_pred==0) )
    #sInCorrectAndLocal = torch.sum( (s_pred != target) * (t_pred==target) * (gate_pred==1) )

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

        th=-0.1
        weights = (   (labels==0) * wt_neg *       (1. + torch.clamp(F.relu(th+ g[:,1]-g[:,0] ), max=0.4) ) * (s_pred!=target) * (t_pred==target) 
                    + (labels==0) * wt_neg * 0.1 * (1. + torch.clamp(F.relu(th+ g[:,1]-g[:,0] ), max=0.4) ) * (s_pred!=target) * (t_pred!=target) 
                    + (labels==1) * wt_pos *       (1. + torch.clamp(F.relu( g[:,0]-g[:,1] -th), max=0.4) )  )


        #weights = ( (labels==0)* wt_neg + (labels==1)*wt_pos )
        #weights = ( (labels==0)*1.2 + (labels==1)*1. )
        #weights = ( (labels==0)*2. + (labels==1)*1. )
        #weights = ( (labels==0)*4. + (labels==1)*1. )
        #weights = ( (labels==0)*5. + (labels==1)*1. )
        '''#weights = ( (labels==0)*2.5 + (labels==1)*1. )
        th=-0.1
        weights = (   (labels==0)* 0.9 * (0.2 + 4.* torch.clamp(F.relu(th+ g[:,1]-g[:,0] ), max=0.8) ) * (s_pred!=target) * (t_pred==target) 
                    + (labels==0)* 0.01 * (0.2 + torch.clamp(F.relu(th+ g[:,1]-g[:,0] ), max=0.2) ) * (s_pred!=target) * (t_pred!=target) 
                    + (labels==1)* 0.2 * (0.1 + 4.* torch.clamp(F.relu( g[:,0]-g[:,1] ), max=0.4) ) 
        ) #'''

        #gate_loss = F.multi_margin_loss( gate, labels, reduction='none' )  
        gate_loss = F.cross_entropy( gate, labels, reduction='none' )  

        clf_loss = torch.mean( weights * gate_loss )

        cov_loss = 100. * F.relu( torch.mean( torch.clamp(F.relu( gate[:,0]-gate[:,1] ), max=0.1) ) - 0.1*args.cov) #0.45 )
        
        #parameters = []
        #for parameter in gating_model.parameters():
        #    parameters.append(parameter.view(-1))
        #l1_loss = 0.1 * torch.abs(torch.cat(parameters)).mean() 

        loss = clf_loss + cov_loss #+ l1_loss
        #print('\t\t[Gating] clf=', clf_loss.item(), ' -- cov=', cov_loss.item()) #, ' l1= ', l1_loss.item())

    elif part=='teacher':
        #weights = (1. * (gate_pred==1) + 1.2 * (gate_pred==0))
        #weights = (1. * (labels==1) + 1.2 * (labels==0))
        weights = (1. * (gate_pred==1) + 1.5 * (gate_pred==0))
        #weights = (1. * (gate_pred==1) + 3. * (gate_pred==0))
        #weights = (1. * (labels==1) + 3. * (labels==0))
        loss = torch.mean( F.cross_entropy(t_logits, target, reduction='none') * weights ) 

        '''temperature = 1.
        log_student = F.log_softmax(t_logits / temperature, dim=1)
        sof_teacher = F.softmax(s_logits / temperature, dim=1)
        loss += torch.mean( torch.sum(F.kl_div(log_student, sof_teacher, reduction="none"), dim=1) * ( alpha * temperature * temperature ) * weights )'''

    elif part=='student':
        #weights = ( 1.2 * (labels==1) * (t_pred==target) + (1. - torch.clamp(s_loss, 0, 0.2)) * (labels==1) * (t_pred!=target) 
        #          + 1. * (labels==0) * (t_pred==target) + (1. - torch.clamp(t_loss, 0, 0.2)) * (labels==0) * (t_pred!=target) )    

        #weights = ( 1.2 * (gate_pred==1) * (t_pred==target) + (1. - torch.clamp(t_loss, 0, 0.2)) * (gate_pred==1) * (t_pred!=target) 
        #          + 1. * (gate_pred==0) * (t_pred==target) + (1. - torch.clamp(t_loss, 0, 0.2)) * (gate_pred==0) * (t_pred!=target) )    
        #weights = (5. * (gate_pred==1) * (t_pred==target) + (1. - torch.clamp(t_loss, 0, 0.5)) * (gate_pred==1) * (t_pred!=target) + 2. * (gate_pred==0))    
        #weights = (1.5 * (gate_pred==1) + 1. * (gate_pred==0))
        #weights = (1. * (gate_pred==1) + 1. * (gate_pred==0))
        #weights = (3. * (labels==1) + 1. * (labels==0))
        #weights = (1.2 * (labels==1) + 1. * (labels==0))
        weights = (1.5 * (gate_pred==1) + 1. * (gate_pred==0))

        s_targets = target #torch.argmax(t_logits, dim=1)  # targets
        #s_targets = torch.argmax(t_logits, dim=1)  # targets

        #loss = torch.mean( F.cross_entropy(s_logits, targets, reduction='none') * (1.0 - alpha) * weights )
        #loss = torch.mean( F.cross_entropy(s_logits, s_targets, reduction='none') * (1.0 - alpha) * weights )
        loss = torch.mean( F.cross_entropy(s_logits, s_targets, reduction='none') *  weights )
        #loss += torch.mean( F.multi_margin_loss(s_logits, s_targets, reduction='none') * (1.0 - alpha) * weights )
        clf_loss = loss

        #log_student = F.log_softmax(s_logits / temperature, dim=1)
        #sof_teacher = F.softmax(t_logits / temperature, dim=1)
        #kl_loss = torch.mean( torch.sum(F.kl_div(log_student, sof_teacher, reduction="none"), dim=1) * ( alpha * temperature * temperature ) * weights )
        #loss = clf_loss + kl_loss
        #print('\t\tclf=', clf_loss.item(), ' -- kl=', kl_loss.item())
        #assert(1==2)

    return loss, s_logits, t_logits, oracle_at_cov, coverage, t_acc_at_cov, acc_at_cov, gate_acc
  


def train(train_loader, model, global_model, routingNet, criterion, optimizer, epoch, args, s_optimizer, t_optimizer):
    if args.n_parts == 3:
        PARTS = ['gating', 'teacher', 'student']
    elif args.n_parts == 2:
        PARTS = ['gating', 'student']
    else:
        PARTS = ['gating']

    part_iters = {}
    part_iters['gating'] = part_iters['teacher'] = part_iters['student'] = 5005

    if hasattr(args, 'part_iters'):
        if args.part_iters is not None:
            part_iters = args.part_iters
    print(part_iters)

    if args.PARTS is not None:
        PARTS = args.PARTS

    print(PARTS)
    #PARTS = ['student']
    for kt in range(args.kt):
      for part in PARTS: 
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        Ttop1 = AverageMeter('TAcc@1', ':6.2f')
        Ttop5 = AverageMeter('TAcc@5', ':6.2f')

        oracleAtCov = AverageMeter('OracleAtCov', ':6.2f')
        tAtCov = AverageMeter('tAtCov', ':6.2f')
        sAtCov = AverageMeter('sAtCov', ':6.2f')
        lcov = AverageMeter('lcov', ':6.2f')
        gateAcc = AverageMeter('gateAcc', ':6.2f')

        progress = ProgressMeter(
            len(train_loader),
            #[batch_time, data_time, losses, top1, top5, Ttop1, Ttop5, lcov, oracleAtCov, tAtCov, sAtCov, gateAcc],
            [batch_time, data_time, losses, top1, Ttop1, lcov, oracleAtCov, tAtCov, sAtCov, gateAcc],
            prefix="[{}] E: [{}]".format(part.upper()[0], epoch))

        # switch to train mode
        model.eval()
        global_model.eval()
        routingNet.eval()
        if part=='gating':
            routingNet.train()
        elif part=='teacher':
            global_model.train()
            if args.global_bn_freeze==1:
                global_model.module.freeze_bn()
        elif part=='student':
            model.train()
            if args.base_bn_freeze==1:
                model.module.freeze_bn()

        end = time.time()

        #for kt in range(args.kt):
        for i, (images, target, g_images, g_target) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            g_images = g_images.cuda(non_blocking=True)
            g_target = g_target.cuda(non_blocking=True)
            #assert( torch.allclose( target, g_target ) ) 

            optimizer.zero_grad()
            s_optimizer.zero_grad()
            t_optimizer.zero_grad()

            loss, s_logits, t_logits, oracle_at_cov, coverage, t_acc_at_cov, acc_at_cov, gate_acc = get_loss( g_images, images, target, model, global_model, routingNet, criterion, args, part=part )


            oracleAtCov.update(oracle_at_cov.item(), images.size(0) )
            if (math.isnan(t_acc_at_cov.item())==False) and (math.isnan(acc_at_cov.item())==False) : 
                lcov.update(coverage.item(), images.size(0) )
                tAtCov.update(t_acc_at_cov.item(), images.size(0) )
                sAtCov.update(acc_at_cov.item(), images.size(0) )
            gateAcc.update( gate_acc.item(), images.size(0) )

            # measure accuracy and record loss
            acc1, acc5 = accuracy(s_logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            acc1, acc5 = accuracy(t_logits, target, topk=(1, 5))
            Ttop1.update(acc1.item(), images.size(0))
            Ttop5.update(acc5.item(), images.size(0))

            # compute gradient and do SGD step
            loss.backward()
            if part=='gating':
                optimizer.step()
            elif part=='student':
                s_optimizer.step()
            elif part=='teacher':
                t_optimizer.step()
            else:
                assert(1==2)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                #torch.cuda.empty_cache()
                progress.display(i)
                #if i>50: break
                if i>part_iters[part]: break


def adjust_bn_stats( train_loader, model, global_model, args ):
    model.train()
    global_model.train()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, g_images, g_target) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            g_images = g_images.cuda(non_blocking=True)
            g_target = g_target.cuda(non_blocking=True)

            model( images )
            global_model( g_images )

            if i%100 == 0: print('iters = ', i)
            #if i > 200: break
 

def validate(val_loader, model, global_model, routingNet, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    Ttop1 = AverageMeter('TAcc@1', ':6.2f')
    Ttop5 = AverageMeter('TAcc@5', ':6.2f')

    oracleAtCov = AverageMeter('OracleAtCov', ':6.2f')
    tAtCov = AverageMeter('tAtCov', ':6.2f')
    sAtCov = AverageMeter('sAtCov', ':6.2f')
    lcov = AverageMeter('lcov', ':6.2f')
    gateAcc = AverageMeter('gateAcc', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, Ttop1, Ttop5, lcov, oracleAtCov, tAtCov, sAtCov, gateAcc],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    global_model.eval()
    routingNet.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, g_images, g_target) in enumerate(val_loader):
        #for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            g_images = g_images.cuda(non_blocking=True)
            g_target = g_target.cuda(non_blocking=True)
            #assert( torch.allclose( target, g_target ) ) 

            loss, s_logits, t_logits, oracle_at_cov, coverage, t_acc_at_cov, acc_at_cov, gate_acc = get_loss( g_images, images, target, model, global_model, routingNet, criterion, args, part='student' )

            oracleAtCov.update(oracle_at_cov.item(), images.size(0) )
            if (math.isnan(t_acc_at_cov.item())==False) and (math.isnan(acc_at_cov.item())==False) : 
                lcov.update(coverage.item(), images.size(0) )
                tAtCov.update(t_acc_at_cov.item(), images.size(0) )
                sAtCov.update(acc_at_cov.item(), images.size(0) )
            gateAcc.update( gate_acc.item(), images.size(0) )

            # measure accuracy and record loss
            acc1, acc5 = accuracy(s_logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            acc1, acc5 = accuracy(t_logits, target, topk=(1, 5))
            Ttop1.update(acc1.item(), images.size(0))
            Ttop5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                #torch.cuda.empty_cache()
                progress.display(i)

        #hybrid_flops = 57 + (1. - lcov.avg) * 215
        hybrid_flops = args.base_flops + (1. - lcov.avg) * args.global_flops
        hybrid_acc = ( lcov.avg * sAtCov.avg + (1. - lcov.avg) * tAtCov.avg ) * 100
        print( 'base_flops=', args.base_flops, ' -- global_flops=', args.global_flops, ' -- lcov=', lcov.avg )
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}  Hybrid@1 {htop1:.3f}  Flops {flops:.3f} --- TAcc@1 {Ttop1.avg:.3f}  TAcc@5 {Ttop5.avg:.3f} '
              .format(top1=top1, top5=top5, htop1=hybrid_acc, flops=hybrid_flops, Ttop1=Ttop1, Ttop5=Ttop5))

    return hybrid_acc #top1.avg

def get_model_prefix( args ):
    prefix = './models/' \
            + args.base_type + '-' + args.base_arch + '-' \
            + args.global_type + '-' + args.global_arch + '-' \
            + args.routing_arch + '-trn-' \
            + str(args.n_parts)+'-' 
    return prefix

def get_trainable_params(model):
    var = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        var.append( param )
    return var

def get_optimizer( lr, optim_type, args, model, wd = None ):
    use_wd = args.wd
    if wd is not None:
      use_wd = wd
    model_vars = get_trainable_params(model)
    print('len models before freeze --', len(model_vars), ' -- ', optim_type, ' lr=', lr, ' -- wd=', use_wd, ' --mom=', args.momentum)
    if optim_type == 'sgd':
        optimizer = torch.optim.SGD(model_vars, lr, momentum=args.momentum, weight_decay=use_wd)
    elif optim_type == 'asgd':
        optimizer = torch.optim.ASGD(model_vars, lr, t0=0.)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model_vars, lr, weight_decay=use_wd)
    else:
        optimizer = torch.optim.Adam(model_vars, lr, weight_decay=use_wd)
    return optimizer

def boostrap_valid(model, val_loader, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Bootstrap Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logits, _ = model(images)
            loss = F.cross_entropy( logits, target )

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                #torch.cuda.empty_cache()
                progress.display(i)

    print(' Bootstrap * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

def bootstrap_train(model, loader, optimizer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1, top5],
        prefix='Bootstrap Train: ')

    # switch to evaluate mode
    #model.train()
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            optimizer.zero_grad()

            logits, _ = model(images)
            loss = F.cross_entropy( logits, target )

            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                #torch.cuda.empty_cache()
                progress.display(i)

    print(' Bootstrap * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg

def bootstrap_model(lr, optim_type, args, model, train_loader, val_loader, start_epoch=0, epochs=5):
    best_acc1 = boostrap_valid(model, val_loader, args)

    optimizer = get_optimizer( lr, optim_type, args, model )
    for epoch in range(start_epoch, epochs):
        bootstrap_train(model, train_loader, optimizer, args)
        acc1 = boostrap_valid(model, val_loader, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'base_state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, prefix=get_model_prefix( args ) + '-bootstrap') #'./models/' + base_model_name+'-'+global_model_name+'-'+str(args.n_parts)+'-' )

def get_mlr(lr_scheduler):
     return lr_scheduler.optimizer.param_groups[0]['lr']

def main_train_eval_loop( args, model, global_model, routingNet, 
      joint_train_loader, joint_val_loader, train_loader, val_loader, global_train_loader, global_val_loader, 
      base_model_name, global_model_name, resume_checkpoint=None,
      start_epoch=0, epochs=40, s_lr=0.001, t_lr=0.001, g_lr=0.01, steps = 10):

    #adjust_bn_stats( joint_train_loader, model, global_model, args )

    print('len models before freeze --', len(list(model.parameters())))
    print('len Global models before freeze --', len(list(global_model.parameters())))
    if args.global_backbone_freeze==1:
        global_model.module.freeze_backbone()
    if args.base_backbone_freeze==1:
        model.module.freeze_backbone()

    if args.base_bootstrap:
        bootstrap_model(args.s_lr, args.base_opt_type, args, model, train_loader, val_loader, start_epoch=0, epochs=5)
    if args.global_bootstrap:
        bootstrap_model(args.t_lr, args.global_opt_type, args, global_model, global_train_loader, global_val_loader, start_epoch=0, epochs=5)

    criterion = nn.CrossEntropyLoss().cuda()
    s_optimizer = get_optimizer( args.s_lr, args.base_opt_type, args, model )
    t_optimizer = get_optimizer( args.t_lr, args.global_opt_type, args, global_model )
    optimizer = get_optimizer( args.g_lr, args.routing_opt_type, args, routingNet, wd=args.r_wd )

    steps = epochs
    if args.routing_use_scheduler:
        scheduler = CosineLRScheduler( optimizer, t_initial=epochs, lr_min=args.min_lr, warmup_lr_init=args.warmup_lr, warmup_t=args.warmup_t )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    if args.base_use_scheduler:
        s_scheduler = CosineLRScheduler( s_optimizer, t_initial=epochs, lr_min=args.min_lr, warmup_lr_init=args.warmup_lr, warmup_t=args.warmup_t )
    else:
        s_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(s_optimizer, steps)

    if args.global_use_scheduler:
        t_scheduler = CosineLRScheduler( t_optimizer, t_initial=epochs, lr_min=args.min_lr, warmup_lr_init=args.warmup_lr, warmup_t=args.warmup_t )
    else:
        t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t_optimizer, steps)

    best_acc1 = validate(joint_val_loader, model, global_model, routingNet, criterion, args)

    #return
    #assert(1==2)

    if resume_checkpoint is not None:
        base_checkpoint = torch.load( resume_checkpoint )
        load_all_models( base_checkpoint, model, global_model, routingNet )
        optimizer.load_state_dict(base_checkpoint['optimizer'])
        s_optimizer.load_state_dict(base_checkpoint['s_optimizer'])
        t_optimizer.load_state_dict(base_checkpoint['t_optimizer'])
        scheduler.load_state_dict(base_checkpoint['scheduler'])
        s_scheduler.load_state_dict(base_checkpoint['s_scheduler'])
        t_scheduler.load_state_dict(base_checkpoint['t_scheduler'])
        start_epoch = base_checkpoint['epoch']
        best_acc1 = base_checkpoint['best_acc1']
        cur_acc1 = validate(joint_val_loader, model, global_model, routingNet, criterion, args)

    #num_epochs = lr_scheduler.get_cycle_length()

    #best_acc1 = 67.1 
    for epoch in range(start_epoch, epochs):
        #torch.cuda.empty_cache()

        # train for one epoch
        train(joint_train_loader, model, global_model, routingNet, criterion, optimizer, epoch, args, s_optimizer, t_optimizer)
        #model_ema.update(model)  

        # evaluate on validation set
        acc1 = validate(joint_val_loader, model, global_model, routingNet, criterion, args)
        #ema_acc1 = validate(val_loader, model_ema, global_model, routingNet, criterion, args)

        if args.routing_use_scheduler: scheduler.step(epoch)
        if args.base_use_scheduler: s_scheduler.step(epoch)
        if args.global_use_scheduler: t_scheduler.step(epoch)
        #print('\t\t LR=', scheduler.get_lr(), ' -- base-LR=', s_scheduler.get_lr(), ' -- global-LR=', t_scheduler.get_lr())
        #print('\t\t LR=', scheduler._get_lr(epoch), ' -- base-LR=', s_scheduler._get_lr(epoch), ' -- global-LR=', t_scheduler._get_lr(epoch))
        print('\t\t LR=', get_mlr(scheduler), ' -- base-LR=', get_mlr(s_scheduler), ' -- global-LR=', get_mlr(t_scheduler))

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'base_state_dict': model.state_dict(),
                #'base_ema_state_dict': model_ema.state_dict(),
                'global_state_dict': global_model.state_dict(),
                'routing_state_dict': routingNet.state_dict(),
                'best_acc1': best_acc1,

                'scheduler' : scheduler.state_dict(),
                's_scheduler' : s_scheduler.state_dict(),
                't_scheduler' : t_scheduler.state_dict(),

                'optimizer' : optimizer.state_dict(),
                't_optimizer' : t_optimizer.state_dict(),
                's_optimizer' : s_optimizer.state_dict(),
            }, is_best, prefix=get_model_prefix( args )) #'./models/' + base_model_name+'-'+global_model_name+'-'+str(args.n_parts)+'-' )


def get_flops_params( model, resolution, label='Base' ):
    total_ops = count_net_flops(model, [1, 3, resolution, resolution])
    total_params = sum([p.numel() for p in model.parameters()])
    print(' * ['+label+'] FLOPs: {:.4}M, param: {:.4}M'.format(total_ops / 1e6, total_params / 1e6))
    x, ft = model( torch.randn(1,3, resolution, resolution) )
    print('Dummy output model --> x ft', x.size(), ft.size())
    num_ft = ft.size()[-1]
    return total_ops, total_params, num_ft

def get_model_config( mean=None, std=None, resolution=224, pct=0.875 ):
    if mean is None:
        mean=[0.485, 0.456, 0.406]
    if std is None:
        std=[0.229, 0.224, 0.225]
    return {
        'num_classes': 1000, 'input_size': (3, resolution, resolution), 'pool_size': (1, 1),
        'crop_pct': pct, 'interpolation': 'bilinear',
        'mean': mean, 'std': std,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
    }

def create_model_from_type( model_type, model_name, config=None, net=None, reset_stats=False, args=None ):

    ofa_config_map = {}
    ofa_config_map['mcunet_base_fixed_global_search'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [7, 3, 7, 3, 5, 3, 7, 3, 5, 7, 3, 5, 3, 5, 5, 3, 3, 5, 5, 3],
       'e': [3, 4, 6, 3, 4, 4, 4, 3, 4, 4, 3, 4, 6, 4, 4, 6, 6, 6, 6, 4],
       'd': [2, 3, 3, 3, 3],
       'r': 176
    }

    ofa_config_map['re4_ofa_150M_constrained_base'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [7, 5, 7, 5, 5, 3, 7, 5, 5, 7, 5, 3, 3, 5, 3, 3, 3, 3, 3, 5], 
      'e': [3, 4, 3, 3, 4, 4, 3, 6, 4, 3, 3, 4, 4, 4, 3, 6, 4, 4, 6, 4], 
      'd': [2, 2, 4, 3, 3], 
      'r': 144
    }
    ofa_config_map['re4_ofa_150M_constrained_global'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [3, 5, 7, 3, 3, 5, 3, 3, 7, 3, 3, 3, 7, 5, 7, 3, 7, 7, 5, 3], 
      'e': [6, 6, 6, 4, 3, 4, 6, 3, 3, 3, 4, 6, 6, 6, 4, 4, 6, 4, 3, 4], 
      'd': [2, 3, 3, 3, 3],
      'r': 192
    }


    ofa_config_map['re3_ofa_150M_constrained_base'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [7, 3, 7, 5, 5, 5, 7, 7, 7, 3, 3, 7, 3, 5, 5, 5, 3, 3, 3, 3], 
      'e': [6, 3, 6, 6, 4, 3, 4, 4, 4, 4, 4, 3, 4, 3, 4, 6, 6, 4, 4, 4], 
      'd': [2, 2, 2, 2, 4],
      'r': 144
    }
    ofa_config_map['re3_ofa_150M_constrained_global'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [7, 3, 7, 5, 7, 3, 5, 3, 5, 3, 5, 3, 7, 3, 7, 7, 7, 3, 7, 3], 
      'e': [3, 6, 4, 6, 6, 4, 6, 3, 6, 3, 4, 6, 4, 4, 3, 3, 3, 4, 6, 6],
      'd': [2, 4, 4, 4, 3],
      'r': 192
    }


    ofa_config_map['f2_ofa_150M_constrained_base'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [3, 3, 7, 5, 7, 3, 7, 5, 5, 7, 5, 3, 3, 5, 3, 3, 3, 3, 3, 5], 
      'e': [3, 4, 3, 3, 3, 3, 3, 6, 4, 3, 3, 3, 4, 4, 3, 6, 4, 4, 4, 4], 
      'd': [2, 2, 2, 3, 3],
      'r': 144
    }
    ofa_config_map['f2_ofa_150M_constrained_global'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [5, 5, 3, 5, 7, 3, 5, 3, 5, 3, 5, 3, 7, 5, 5, 7, 5, 7, 7, 3], 
      'e': [4, 4, 3, 4, 6, 4, 3, 6, 3, 3, 4, 6, 4, 6, 6, 3, 6, 4, 6, 6], 
      'd': [2, 4, 4, 4, 4],
      'r': 192
    }

    ofa_config_map['f_ofa_150M_constrained_base'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [3, 3, 7, 5, 7, 3, 7, 5, 5, 7, 5, 3, 3, 5, 3, 3, 3, 3, 3, 5], 
      'e': [3, 4, 3, 3, 3, 3, 3, 6, 4, 3, 3, 3, 4, 4, 3, 6, 4, 4, 4, 4], 
      'd': [2, 2, 2, 3, 3],
      'r': 144
    }
    ofa_config_map['f_ofa_150M_constrained_global'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [5, 3, 3, 5, 7, 3, 5, 3, 5, 3, 5, 3, 7, 5, 5, 7, 5, 3, 7, 3], 
      'e': [3, 6, 6, 6, 6, 4, 6, 3, 3, 3, 4, 6, 4, 6, 6, 3, 6, 4, 6, 6], 
      'd': [2, 4, 4, 4, 4],
      'r': 192
    }

    ofa_config_map['re2_ofa_150M_constrained_base'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [3, 3, 7, 5, 3, 3, 7, 5, 7, 7, 3, 7, 3, 7, 7, 7, 3, 3, 7, 3], 
      'e': [3, 3, 6, 6, 3, 3, 6, 4, 4, 4, 3, 3, 4, 3, 3, 4, 4, 4, 4, 4],
      'd': [2, 2, 2, 2, 3],
      'r': 144
    }
    ofa_config_map['re2_ofa_150M_constrained_global'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [7, 3, 3, 5, 7, 7, 3, 3, 5, 5, 5, 7, 7, 3, 7, 5, 7, 7, 7, 3], 
      'e': [3, 6, 3, 4, 4, 4, 4, 3, 6, 6, 4, 6, 6, 4, 4, 4, 6, 6, 6, 6], 
      'd': [4, 4, 3, 4, 4], 
      'r': 192
    }

    ofa_config_map['re_ofa_150M_constrained_base'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [3, 3, 7, 5, 7, 3, 7, 5, 7, 3, 3, 7, 3, 5, 3, 3, 3, 3, 3, 3], 
      'e': [3, 3, 6, 6, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 6, 6, 4, 4, 4],
      'd': [2, 2, 2, 2, 3], 
      'r': 144
    }

    ofa_config_map['re_ofa_150M_constrained_global'] = {
      'net': 'ofa_mbv3_d234_e346_k357_w1.0',
      'ks': [7, 3, 3, 5, 7, 3, 5, 3, 5, 5, 5, 5, 5, 3, 5, 7, 5, 7, 7, 3], 
      'e': [3, 6, 3, 4, 6, 6, 6, 4, 3, 4, 4, 6, 4, 4, 4, 3, 3, 3, 6, 6], 
      'd': [2, 4, 4, 4, 4], 
      'r': 192
    }


    ofa_config_map['re2_ofa_250M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 7, 3, 7, 3], 
       'e': [3, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 6, 3, 4, 3, 3, 3, 3], 
       'd': [2, 2, 2, 2, 2],
       'r': 128
    }
    ofa_config_map['re2_ofa_250M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 7, 7, 3, 5, 7, 5, 7, 5, 5, 5, 3, 7, 7, 3, 5, 5, 7, 5, 7], 
       'e': [3, 6, 3, 4, 6, 4, 4, 3, 4, 4, 6, 6, 4, 6, 6, 3, 6, 6, 4, 6], 
       'd': [2, 4, 4, 4, 4],
       'r': 224
    }


    ofa_config_map['re3_ofa_250M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [5, 5, 3, 5, 7, 5, 3, 7, 7, 3, 3, 5, 5, 7, 5, 5, 5, 7, 5, 3], 
       'e': [4, 4, 6, 3, 6, 4, 3, 4, 4, 4, 4, 3, 6, 4, 4, 4, 4, 4, 6, 3], 
       'd': [2, 2, 2, 2, 4],
       'r': 144,
    }
    ofa_config_map['re3_ofa_250M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [7, 5, 3, 5, 7, 5, 7, 5, 5, 5, 5, 3, 3, 3, 5, 7, 7, 7, 3, 3], 
       'e': [4, 4, 4, 4, 6, 4, 4, 6, 4, 4, 6, 6, 6, 4, 6, 4, 4, 6, 4, 4], 
       'd': [4, 3, 3, 4, 4],
       'r': 208
    }


    ofa_config_map['re_ofa_250M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 3, 3, 7, 5, 5, 7, 3, 7, 5, 3, 5, 3, 7, 3, 5, 3, 3, 5, 3], 
       'e': [4, 4, 6, 3, 3, 4, 3, 6, 6, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 3],
       'd': [2, 3, 4, 4, 4],
       'r': 144
    }
    ofa_config_map['re_ofa_250M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [7, 3, 7, 3, 7, 7, 7, 3, 5, 5, 5, 5, 3, 3, 5, 7, 5, 7, 7, 3], 
       'e': [4, 6, 3, 4, 3, 4, 4, 4, 6, 4, 6, 6, 6, 4, 6, 4, 3, 6, 6, 4], 
       'd': [4, 2, 3, 4, 2],
       'r': 192
    }


    ofa_config_map['ofa_150M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 3, 3, 3, 5, 3, 3, 3, 7, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 3],
       'e': [3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3],
       'd': [2, 2, 2, 2, 2],
       'r': 128
    }
    ofa_config_map['ofa_150M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [7, 3, 7, 5, 5, 7, 3, 7, 7, 5, 5, 5, 3, 3, 7, 3, 5, 7, 3, 5],
       'e': [3, 3, 3, 6, 4, 4, 6, 4, 6, 4, 4, 6, 4, 4, 4, 6, 4, 6, 6, 4],
       'd': [3, 3, 3, 3, 4],
       'r': 176
    }

    ofa_config_map['ofa_250M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 7, 3, 7, 3],
       'e': [3, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 6, 3, 4, 3, 3, 3, 3],
       'd': [2, 2, 2, 2, 2],
       'r': 128
    }
    ofa_config_map['ofa_250M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 7, 7, 3, 5, 7, 5, 7, 5, 5, 5, 3, 7, 7, 3, 5, 5, 7, 5, 7],
       'e': [3, 6, 3, 4, 6, 4, 4, 3, 4, 4, 6, 6, 4, 6, 6, 3, 6, 6, 4, 6],
       'd': [2, 4, 4, 4, 4],
       'r': 224
    }
    ofa_config_map['ofa_350M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 7, 7, 3, 3, 5, 3, 3, 3, 3, 3, 7, 3, 5, 3, 3, 3, 3, 3, 5],
       'e': [3, 6, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 6, 3, 3, 3, 3, 4, 3, 3],
       'd': [3, 2, 2, 2, 2],
       'r': 176
    }
    ofa_config_map['ofa_350M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [5, 5, 5, 7, 5, 5, 7, 7, 5, 5, 3, 3, 7, 7, 5, 7, 7, 5, 3, 7],
       'e': [3, 4, 4, 4, 4, 6, 6, 3, 4, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 4],
       'd': [4, 3, 3, 4, 4],
       'r': 208
    }

    ofa_config_map['re_ofa_350M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [5, 5, 3, 5, 5, 7, 7, 3, 7, 7, 5, 3, 3, 7, 3, 7, 3, 7, 5, 3], 
       'e': [4, 3, 4, 3, 6, 4, 6, 4, 4, 6, 6, 6, 3, 4, 6, 4, 6, 6, 6, 3],
       'd': [4, 4, 2, 4, 4], 
       'r': 144
    }
    ofa_config_map['re_ofa_350M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [7, 3, 7, 3, 7, 7, 3, 3, 5, 3, 5, 7, 7, 7, 5, 5, 5, 5, 3, 7], 
       'e': [3, 4, 3, 6, 3, 4, 4, 4, 3, 4, 6, 6, 4, 6, 6, 4, 6, 4, 6, 6],
       'd': [3, 4, 3, 4, 4],
       'r': 224
    }


    ofa_config_map['re2_ofa_350M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [5, 5, 5, 5, 5, 7, 7, 5, 7, 7, 3, 7, 3, 7, 3, 7, 7, 7, 7, 5], 
       'e': [4, 3, 3, 3, 4, 4, 6, 3, 4, 6, 3, 6, 3, 4, 6, 4, 6, 6, 3, 6],
       'd': [4, 4, 4, 4, 4], 
       'r': 144
    }
    ofa_config_map['re2_ofa_350M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [5, 7, 7, 5, 7, 7, 7, 3, 5, 3, 5, 7, 7, 7, 5, 5, 5, 7, 3, 5], 
       'e': [4, 4, 3, 3, 6, 4, 4, 4, 3, 4, 6, 6, 4, 6, 6, 4, 6, 6, 6, 4],
       'd': [3, 3, 3, 4, 4], 
       'r': 224
    }


    ofa_config_map['re_ofa_400M_constrained_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       'ks': [3, 5, 3, 3, 5, 3, 3, 3, 5, 5, 3, 3, 7, 5, 3, 3, 3, 3, 7, 5], 
       'e': [3, 4, 3, 3, 3, 4, 6, 3, 4, 4, 3, 3, 4, 4, 4, 4, 6, 6, 6, 3], 
       'd': [2, 3, 2, 4, 4],
       'r': 208
    }
    ofa_config_map['re_ofa_400M_constrained_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.0',
       'ks': [5, 3, 3, 3, 5, 3, 5, 3, 5, 5, 3, 3, 5, 7, 3, 3, 7, 3, 3, 3], 
       'e': [4, 3, 3, 3, 4, 4, 4, 3, 4, 6, 3, 3, 6, 6, 6, 6, 6, 6, 6, 3], 
       'd': [3, 4, 4, 3, 4],
       'r': 224
    }


    ofa_config_map['re_ofa_150M_base_cov_search_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.0',

					'ks': [3, 3, 3, 5, 5, 7, 5, 5, 5, 5, 5, 5, 7, 7, 5, 5, 7, 7, 5, 5],
					'e': [4, 3, 6, 6, 4, 6, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 4],
					'd': [2, 3, 2, 4, 4],

       "r": 144
    }



    ofa_config_map['ofa_150M_base_cov_search_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       "ks": [3, 7, 5, 3, 5, 3, 3, 3, 7, 7, 5, 7, 7, 3, 5, 7, 3, 5, 5, 5], 
       "e": [3, 4, 4, 3, 3, 6, 3, 3, 4, 4, 6, 3, 6, 6, 3, 4, 3, 4, 6, 6], 
       "d": [2, 3, 2, 4, 2], 
       "r": 144
    }

    ofa_config_map['ofa_150M_base_cov_search_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       "ks": [5, 5, 5, 7, 3, 7, 3, 5, 5, 5, 7, 7, 3, 7, 3, 5, 7, 5, 7, 7], 
       "e": [6, 6, 3, 6, 4, 4, 4, 3, 6, 4, 6, 4, 4, 6, 3, 6, 6, 6, 6, 3], 
       "d": [4, 4, 4, 4, 4], 
       "r": 208
    }


    ofa_config_map['ofa_250M_base_cov_search_base'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       "ks": [5, 5, 7, 3, 3, 7, 3, 5, 7, 5, 3, 5, 3, 7, 5, 7, 5, 5, 7, 3], 
       "e": [6, 3, 6, 6, 6, 3, 6, 6, 6, 4, 4, 3, 6, 3, 3, 6, 4, 6, 6, 3], 
       "d": [2, 3, 2, 4, 2], 
       "r": 176
    }

    ofa_config_map['ofa_250M_base_cov_search_global'] = {
       'net': 'ofa_mbv3_d234_e346_k357_w1.2',
       "ks": [7, 3, 7, 7, 3, 5, 5, 3, 5, 3, 7, 3, 5, 5, 5, 5, 3, 3, 7, 7], 
       "e": [6, 4, 3, 4, 3, 3, 3, 4, 6, 6, 6, 3, 6, 4, 6, 6, 6, 6, 6, 3], 
       "d": [4, 3, 4, 4, 4], 
       "r": 208
   }




    if model_type == 'timm':
        model = timm.create_model(model_name, pretrained=True)
        model_cfg = model.default_cfg
        model = TimmModel(model, model_name)
    elif model_type == 'torchvision':
        model_cfg = get_model_config()
        if model_name == 'mobilenetv3_small':
            model = mobilenet_v3_small( pretrained=True )
        else:
            model = mobilenet_v3_large( pretrained=True )
    elif model_type == 'ofa_spec':
        model, image_size = ofa_specialized(net_id=model_name, pretrained=True)
        model_cfg = get_model_config( resolution=image_size )
        model = OFANetModel(model)
    elif model_type == 'ft_ofa_spec':
        model, image_size = ofa_specialized(net_id=model_name, pretrained=True)
        model_cfg = get_model_config( resolution=image_size )
        model = Ft_OFANetModel(model)
    elif model_type == 'ofa':
        assert( model_name in ofa_config_map )
        #assert(config is not None)
        #assert(net is not None)

        cfg = ofa_config_map[model_name] #config 
        net = cfg['net']
        ofa_network = ofa_net(net, pretrained=True)
        ofa_network.set_active_subnet( ks=cfg['ks'], e=cfg['e'], d=cfg['d'] )
        model = ofa_network.get_active_subnet(preserve_weight=True)
        del ofa_network

        model_cfg = get_model_config( resolution=cfg['r'] )

        if reset_stats:
            ImagenetDataProvider.DEFAULT_PATH = args.path
            run_config = ImagenetRunConfig(test_batch_size=args.batch_size, n_worker=args.workers, image_size=cfg['r'])
            run_manager = RunManager('.tmp/eval_subnet', model, run_config, init=False)
            run_manager.reset_running_statistics(net=model)

            #loss, (top1, top5) = run_manager.validate(net=model)
            #print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

        model = OFANetModel(model)
    elif model_type == 'mcunet':
        config = './configs/' + model_name + '.json'
        assert(config is not None)

        mean = std = [0.5, 0.5, 0.5]
        with open(config) as f:
            cfg = json.load(f)

        model_cfg = get_model_config( mean, std, cfg['resolution'] )
        model = ProxylessNASNets.build_from_config(cfg)

        load_from = './configs/' + model_name + '.pth'
        sd = torch.load(load_from, map_location='cpu')
        model.load_state_dict(sd['state_dict'])

        model = MCUNetModel(model)
    else:
        raise NotImplementedError 
    return model, model_cfg

def get_model( model_type, model_name, model_stats, label, device='cuda', config=None, net=None, args=None, base_checkpoint=None, dict_name="base_state_dict" ):

    #model = mobilenet_v3_small(pretrained=True)

    #model = timm.create_model(model_name, pretrained=True)
    #base_model_cfg = model.default_cfg 
    model, base_model_cfg = create_model_from_type( model_type, model_name )
    resolution = base_model_cfg['input_size'][2]
    #print(resolution)

    total_ops, total_params, num_ft = get_flops_params( model, resolution, label )
    model_stats[model_name+'param'] = total_params
    model_stats[model_name+'flop'] = total_ops

    del model
    model, base_model_cfg = create_model_from_type( model_type, model_name, reset_stats=True, args=args )
    base_model_cfg['num_ft'] = num_ft
    #assert(1==2)
    #model = timm.create_model(model_name, pretrained=True)
    model = torch.nn.DataParallel(model)
    if base_checkpoint is not None:
        model.load_state_dict(base_checkpoint[dict_name])
    model = model.to(device)
    return model, base_model_cfg

def load_all_models( base_checkpoint, base_model, global_model, routingNet ):
    routingNet.load_state_dict(base_checkpoint["routing_state_dict"])
    base_model.load_state_dict(base_checkpoint["base_state_dict"])
    global_model.load_state_dict(base_checkpoint["global_state_dict"])

def get_routing_model( base_checkpoint=None, device='cuda', routing_name='', base_model_cfg=None ):
    if routing_name == 'with_ft':
        num_ft = base_model_cfg['num_ft']
        routingNet = RoutingNetworkWithFt( num_ft )
    elif routing_name == 'extra_ft':
        num_ft = base_model_cfg['num_ft']
        routingNet = RoutingNetworkWithExtraFt( num_ft )
    elif routing_name == 'with_logits':
        num_ft = base_model_cfg['num_ft']
        routingNet = RoutingNetworkWithLogits( num_ft )
    elif routing_name == '20no_ft':
        num_ft = base_model_cfg['num_ft']
        routingNet = RoutingNetworkTop20( )
    elif routing_name == '20ft':
        num_ft = base_model_cfg['num_ft']
        routingNet = RoutingNetworkTop20WithFt( num_ft )
    else:
        routingNet = RoutingNetwork()
    routingNet = torch.nn.DataParallel(routingNet)
    routingNet = routingNet.to(device)

    if base_checkpoint is not None:
        routingNet.load_state_dict(base_checkpoint["routing_state_dict"])
    return routingNet

def add_pd_data(pd_data, global_model_name, model_name, prefix,
       global_model_stats, base_model_stats, hybrid_model_stats,
       scheme_name='Entropy', ):

    pd_data.append( [
                  scheme_name,
                  global_model_name, 
                  model_name, 
                  global_model_stats[global_model_name+'flop']/1e6, 
                  global_model_stats[global_model_name+'valid_acc1'],
                  base_model_stats[model_name+'flop']/1e6, 
                  base_model_stats[model_name+'valid_acc1'],
                  hybrid_model_stats[prefix+'_hybrid-cov'],
                  hybrid_model_stats[prefix+'_hybrid-base_pred_acc'],
                  hybrid_model_stats[prefix+'_hybrid-global_abs_acc'],
                  hybrid_model_stats[prefix+'_hybrid-valid_acc'],
                  hybrid_model_stats[prefix+'_hybrid-flop']/1e6,
    ] )

def add_hybrid_stats_in_table( pd_data, base_y_pred, global_y_pred, base_y_true,  base_y_entropy, y_gate_vals,
          model_name, global_model_name, base_model_stats, global_model_stats, hybrid_model_stats, scheme='agreement', cov=0.9 ):
    if scheme == 'agreement':
        scheme_name = 'Oracle-Agreement'
        prefix=model_name+'oracle-agreement'
        route=base_y_pred==global_y_pred
    elif scheme == 'margin-upper':
        scheme_name = 'Oracle-Margin-upper'
        prefix=model_name+'oracle-margin-upper'
        route = base_y_pred==base_y_true #, (global_y_pred!=base_y_true) ) 
    elif scheme == 'margin':
        scheme_name = 'Oracle-Margin'
        prefix=model_name+'oracle-margin'
        route = np.logical_or(base_y_pred==base_y_true, (global_y_pred!=base_y_true) ) 
    elif scheme == 'gate':
        best_thr, _best_cov = get_thr_at_cov(y_gate_vals, target_cov=cov, num=2000, low=-4, high=4)
        #print('[Gating] Best cov = ', _best_cov, ' at thr=', best_thr)
        thr=best_thr #-1.84
        scheme_name = 'Gating-' + '{:.2f}'.format(cov)
        #prefix=model_name+'-gating-vals-'+ str(thr)  + '{:.2f}'.format(cov)
        prefix=model_name+'-gating-vals-'+ '{:.2f}'.format(cov)
        route = ((y_gate_vals[:,1]-y_gate_vals[:,0]) >= thr)*1 
    elif scheme == 'entropy':
        scheme_name = 'Entropy-' + '{:.2f}'.format(cov)
        prefix=model_name+'entropy-' + '{:.2f}'.format(cov)
        best_thr, _best_cov = get_entropy_thr_at_cov(base_y_entropy, target_cov=cov, num=500, low=0, high=5)
        #print('[Entropy] Best cov = ', _best_cov, ' at thr=', best_thr)
        found_th = best_thr
        route=base_y_entropy<=found_th
    else:
        raise NotImplementedError 

    get_hybrid_stats( base_y_pred, global_y_pred, base_y_true, route, np.logical_not(route), 
              base_model_stats, global_model_stats, hybrid_model_stats, prefix=prefix, base_name=model_name, global_name=global_model_name  )
    add_pd_data(pd_data, global_model_name, model_name, prefix, global_model_stats, base_model_stats, hybrid_model_stats, scheme_name=scheme_name, )




