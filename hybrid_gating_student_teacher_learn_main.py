import sys
sys.path.append(".")

import time
from copy import deepcopy
import shutil
import numpy as np
import pandas as pd
#from tabulate import tabulate
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import math
import json

from hybrid_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = 'cuda' # 'cpu' # 

'''
global_model_list = [
    #('torchvision', 'mobilenetv3_small', 224, 0.875),
    ###('mobilenetv3_large_100', 224, 0.875),
    #('mobilenetv3_rw', 224, 0.875),
    ###('tf_mobilenetv3_large_100', 224, 0.875),
    ('timm', 'tf_mobilenetv3_large_075', 224, 0.875),
    ###('tf_mobilenetv3_large_minimal_100', 224, 0.875),
    ('timm', 'tf_mobilenetv3_small_100', 224, 0.875),
    ('timm', 'tf_mobilenetv3_small_075', 224, 0.875),
    #('tf_mobilenetv3_small_minimal_100', 224, 0.875),
    #('', 224),
    #('tf_efficientnet_b0', 224, 0.875),
    #('tf_efficientnet_b1', 256, 0.882),
    #('tf_efficientnet_b2', 288, 0.890),
    #('tf_efficientnet_b3', 320, 0.904),
    #('tf_efficientnet_b4', 384, 0.922),
    #('tf_efficientnet_b5', 456, 0.934),
]'''

'''global_model_list = [
    #('ofa_spec', 'flops@595M_top1@80.0_finetune@75', 224, 0.875),
    ('ofa_spec', 'flops@389M_top1@79.1_finetune@75', 224, 0.875),
    ('ofa_spec', 'LG-G8_lat@24ms_top1@76.4_finetune@25', 224, 0.875),
    ('ofa_spec', 'LG-G8_lat@16ms_top1@74.7_finetune@25', 224, 0.875),
    ('ofa_spec', 'LG-G8_lat@11ms_top1@73.0_finetune@25', 224, 0.875),
    ('ofa_spec', 'LG-G8_lat@8ms_top1@71.1_finetune@25', 224, 0.875),
    ('ofa_spec', 'note8_lat@22ms_top1@70.4_finetune@25', 224, 0.875),
    #('ofa_spec', '', 224, 0.875),
]'''

parser = get_default_parser()
args = parser.parse_args()
print(args)

global_model_list = [ (args.base_type, args.base_arch, 224, 0.875), ]

# type, global-model, base-model, global-flops, global-acc, base-flops, base-acc, base-cov, base@cov, global@cov, hybrid-acc, hybrid-flops
pd_data = []
base_model_stats = {}
global_model_stats = {}
hybrid_model_stats = {}

base_model_cfg = None

global_model_type = args.global_type
global_model_name = args.global_arch
global_model, global_model_cfg = get_model( global_model_type, global_model_name, global_model_stats, 'Global', 
     device, args=args, base_checkpoint=None, dict_name="global_state_dict" )

#'''
joint_train_loader, joint_val_loader, train_loader, val_loader, \
          global_train_loader, global_val_loader = get_base_global_loaders( args.train_dir, args.val_dir, workers=args.workers, 
                global_model_cfg=global_model_cfg, base_model_cfg=base_model_cfg, 
                random_erase_prob=args.random_erase_prob, batch_size=args.batch_size, eval_batch_size=args.batch_size, 
          )

#routingNet = get_routing_model( base_checkpoint, device )


global_loss, global_top1, global_top5, global_y_pred, global_y_true, global_y_entropy, global_y_loss = old_validate(0, global_model, global_val_loader, base=True, args=args, device=device)
global_model_stats[global_model_name+'valid_loss'] = global_loss.item()
global_model_stats[global_model_name+'valid_acc1'] = global_top1.item()
global_model_stats[global_model_name+'valid_acc5'] = global_top5.item() 
#assert(1==2)
    
pd_data.append( [ 'standalone',
                  global_model_name, 
                  '', 
                  global_model_stats[global_model_name+'flop']/1e6, 
                  global_model_stats[global_model_name+'valid_acc1'],
                  0,
                  0,
                  0,
                  0,
                  global_model_stats[global_model_name+'valid_acc1'],
                  global_model_stats[global_model_name+'valid_acc1'],
                  global_model_stats[global_model_name+'flop']/1e6, 
] )

del joint_train_loader, joint_val_loader, train_loader, val_loader, global_train_loader, global_val_loader #'''
#del global_model
torch.cuda.empty_cache()

args.PARTS = None

### Hybrid Models
cnt=0 
for model_type, model_name, resolution, crop_pct in global_model_list:
    print('Current Base-Model = ', model_name)
    model, base_model_cfg = get_model( model_type, model_name, base_model_stats, 'New-Base', device, args=args )
  
    #args.model_ckpt = './models/' + model_name+'-'+global_model_name+'-'+str(args.n_parts) +'-checkpoint.pth.tar'
    args.model_ckpt = get_model_prefix(args)
    base_checkpoint = None 
    if args.routing_arch=='no_ft':
        model_ckpt = './models/mobilenetv3_large_100-efficientnet_b2-checkpoint.pth.tar'
        if os.path.isfile(model_ckpt):
            base_checkpoint = torch.load(model_ckpt) 
        else:
            print('Pre-trained point for routing does not exists.. will initialize randomly..', model_ckpt)
    routingNet = get_routing_model( device=device, routing_name=args.routing_arch, base_model_cfg=base_model_cfg, base_checkpoint=base_checkpoint )


    #''' 
    joint_train_loader, joint_val_loader, train_loader, val_loader, \
          global_train_loader, global_val_loader = get_base_global_loaders( args.train_dir, args.val_dir, workers=args.workers, 
                global_model_cfg=global_model_cfg, base_model_cfg=base_model_cfg, 
                random_erase_prob=args.random_erase_prob, batch_size=args.batch_size, eval_batch_size=args.batch_size, 
          )

    base_loss, base_top1, base_top5, base_y_pred, base_y_true, base_y_entropy, base_y_gate, base_y_loss, y_gate_vals = evaluate_routing_model( routingNet, model, val_loader, args.batch_size, base=True ) 
    base_model_stats[model_name+'valid_acc1'] = base_top1.item()
    base_model_stats[model_name+'valid_acc1-old'] = base_top1.item()

    for scheme in ['agreement', 'margin', 'margin-upper']:
        add_hybrid_stats_in_table( pd_data, base_y_pred, global_y_pred, base_y_true,  base_y_entropy, y_gate_vals,
          model_name, global_model_name, base_model_stats, global_model_stats, hybrid_model_stats, scheme=scheme, )

    for scheme in ['entropy']:
      for cov in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        add_hybrid_stats_in_table( pd_data, base_y_pred, global_y_pred, base_y_true,  base_y_entropy, y_gate_vals,
          model_name, global_model_name, base_model_stats, global_model_stats, hybrid_model_stats, scheme=scheme, cov=cov )

    if args.resume:
        base_checkpoint = torch.load(args.load_from)
        load_all_models( base_checkpoint, model, global_model, routingNet )


    args.base_flops = base_model_stats[model_name+'flop']
    args.global_flops = global_model_stats[global_model_name+'flop']

    g_lr = args.g_lr #0.0004
    t_lr = args.t_lr #0.0001
    s_lr = args.s_lr #0.0001

    part_iters = {}
    part_iters['gating']  = args.g_iters #500
    part_iters['teacher'] = args.t_iters #5005
    part_iters['student'] = args.s_iters #5005 #5005

    args.part_iters = part_iters
    #args.cov = args.cov #0.55 #0.45
    #args.PARTS = ['gating', 'teacher', 'student']
    #args.PARTS = ['gating', 'student']
    #args.n_parts = 2 #1
    epochs = args.epochs #20 #100 #10 #1 # 40
    if not args.eval:
        print('Training this base+global combination.. ', model_name, global_model_name, ' -- epochs=', epochs)
        resume_checkpoint = None
        if (args.load_from is not None ) or args.load_from  != '':
            resume_checkpoint = args.load_from
        main_train_eval_loop( args, model, global_model, routingNet, 
          joint_train_loader, joint_val_loader, train_loader, val_loader, global_train_loader, global_val_loader, 
          model_name, global_model_name, resume_checkpoint=resume_checkpoint, 
          start_epoch=0, epochs=epochs, s_lr=s_lr, t_lr=t_lr, g_lr=g_lr, steps = 10)
    else: 
        print('Will evaluate model now.. ')


    global_loss, global_top1, global_top5, global_y_pred, global_y_true, global_y_entropy, global_y_loss = old_validate(0, global_model, global_val_loader, base=True, args=args, device=device)
    global_model_stats[global_model_name+'valid_loss'] = global_loss.item()
    global_model_stats[global_model_name+'valid_acc1'] = global_top1.item()
    global_model_stats[global_model_name+'valid_acc5'] = global_top5.item() 

    base_loss, base_top1, base_top5, base_y_pred, base_y_true, base_y_entropy, base_y_gate, base_y_loss, y_gate_vals = evaluate_routing_model( routingNet, model, val_loader, args.batch_size, base=True ) 
    base_model_stats[model_name+'valid_loss'] = base_loss.item()
    base_model_stats[model_name+'valid_acc1'] = base_top1.item()
    base_model_stats[model_name+'valid_acc5'] = base_top5.item()

    print(' * Base Accuracy: {:.2f}%'.format(base_top1))
    print(' * Global Accuracy: {:.2f}%'.format(global_top1))
    
    del joint_train_loader, joint_val_loader, train_loader, val_loader, global_train_loader, global_val_loader 
    del model
    torch.cuda.empty_cache()

    for scheme in ['gate']:
      for cov in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        add_hybrid_stats_in_table( pd_data, base_y_pred, global_y_pred, base_y_true,  base_y_entropy, y_gate_vals,
          model_name, global_model_name, base_model_stats, global_model_stats, hybrid_model_stats, scheme=scheme, cov=cov )

    cnt += 1
    #if cnt >= 1: #4: 
    #    break #'''
    
stats_dict = [ global_model_stats, base_model_stats, hybrid_model_stats ]
stats_dict_name = [ 'Global ', 'Base ', 'Hybrid ' ]

for idx in range(len( stats_dict )):
    print( stats_dict_name[idx] ) 
    print( json.dumps( stats_dict[idx], indent=4, sort_keys=True ) )

df = pd.DataFrame( pd_data, columns=['type', 'global-model', 'base-model', 'global-flops', 'global-acc', 'base-flops', 'base-acc', 'base-cov', 'base@cov', 'global@cov', 'hybrid-acc', 'hybrid-flops'] )
print(df.to_markdown()) 

