'''
Copyright 2023 Anil Kag (https://anilkagak2.github.io)

Wrapper on top of timm models for ease in accessing intermediate features and classifier logits
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from thop import profile

def efficientnet_forward_head(model, x):
    x = model.global_pool(x)
    if model.drop_rate > 0.:
        x = F.dropout(x, p=model.drop_rate, training=model.training)
    ft = x
    x = model.classifier(x)
    return x, ft

def mbv3_forward_head(model, x):
    x = model.global_pool(x)
    x = model.conv_head(x)
    x = model.act2(x)
    x = model.flatten(x)
    if model.drop_rate > 0.:
        x = F.dropout(x, p=model.drop_rate, training=model.training)
    ft = x
    x = model.classifier(x)
    return x, ft

class TimmModel(nn.Module):
    def __init__( self, args, model_name, in_chans=3 ):
        super(TimmModel, self).__init__()

        self.model = create_model(
          model_name,
          pretrained=args.pretrained,
          in_chans=in_chans,
          num_classes=args.num_classes,
          drop_rate=args.drop,
          drop_path_rate=args.drop_path,
          drop_block_rate=args.drop_block,
          global_pool=args.gp,
          bn_momentum=args.bn_momentum,
          bn_eps=args.bn_eps,
          scriptable=args.torchscript,
          checkpoint_path=args.initial_checkpoint,
          **args.model_kwargs,
        )

        if 'mobilenetv3' in model_name:
            self.forward_head = mbv3_forward_head
        elif 'efficientnet' in model_name:
            self.forward_head = efficientnet_forward_head
        else:
            raise NotImplementedError

        self.default_cfg = self.model.default_cfg
        self.num_classes = self.model.num_classes
        self.num_features = self.model.num_features

    def forward(self, x):
        ft = self.model.forward_features(x)
        x, ft = self.forward_head(self.model, ft)
        #x = self.model.forward_head(ft)
        return x, ft


def get_model_from_name( args, model_name, model_type='timm' ):
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    if model_type == 'timm':
        model = TimmModel( args, model_name, in_chans )
        model_cfg = model.default_cfg
    else:
        print( 'model_type ' + model_type + ' not supported!' )
        raise NotImplementedError 
    return model, model_cfg, in_chans

def get_model_infos( model, shape, cuda=True ): 
    inputs = torch.randn( *shape )
    if cuda: inputs = inputs.cuda()

    macs, params = profile( model, inputs=(inputs,), verbose=False )
    #macs = macs / 1e6
    #params = params / 2 ** 20
    return macs, params

def get_model_with_stats( args, model_name, model_type='timm' ):
    model, cfg, in_chans = get_model_from_name( args, model_name, model_type )
    resolution = cfg['input_size'][2]
    macs, params = get_model_infos( model, (1, in_chans, resolution, resolution), cuda=False )

    model_stats = {}
    model_stats[model_name+'param'] = params
    model_stats[model_name+'flop'] = macs

    del model
    model, _, _ = get_model_from_name( args, model_name, model_type )
    return model, model_stats

def execute_network( student, teacher, disk_router, hybrid_router, inputs, g_inputs, args, train=False ):
    with torch.no_grad():
        t_logits, t_ft = teacher(g_inputs)

    if train:
      s_logits, s_ft = student(inputs)
      hybrid_gate = hybrid_router( s_logits, s_ft )
      disk_gate = disk_router( t_logits, t_ft )
    else:
      with torch.no_grad():
        s_logits, s_ft = student(inputs)
        hybrid_gate = hybrid_router( s_logits, s_ft )
        disk_gate = disk_router( t_logits, t_ft )

    return s_logits, t_logits, hybrid_gate, disk_gate

