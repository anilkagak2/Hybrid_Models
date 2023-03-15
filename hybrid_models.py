
import json
import torch.nn.functional as F
import torch
import torch.nn as nn

'''
from tinynas.nn.networks import ProxylessNASNets
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
from ofa.model_zoo import ofa_net, ofa_specialized
from ofa.utils.layers import set_layer_from_config, MBConvLayer, ConvLayer, IdentityLayer, LinearLayer, ResidualBlock
from ofa.utils import MyNetwork, make_divisible, MyGlobalAvgPool2d
'''

from timm.models import create_model

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
        self.num_classes = model.num_classes
        self.num_features = model.num_features

    def forward(self, x):
        x = self.model.forward_features(x)
        
        if self.is_efficientnet:
            x = self.global_pool(x)
        else:
            x = self.flatten(x)

        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.model.training)

        ft = x
        x = self.classifier(x)
        return x, ft

class OFABranchyNetModel(nn.Module):
    def __init__(self, model):
        super(OFABranchyNetModel, self).__init__()

        self.num_classes = 1000 #model.num_classes
        self.model = model
        self.first_conv = model.first_conv
        self.blocks = model.blocks
        self.final_expand_layer = model.final_expand_layer
        self.global_avg_pool = model.global_avg_pool
        self.feature_mix_layer = model.feature_mix_layer
        self.classifier = model.classifier
        self.skip_len = 2

    def forward(self, x):
        x = self.first_conv(x)
        first_block = None
        for block in self.blocks:
            x = block(x)
            if first_block is None:
                first_block = x
        x = self.final_expand_layer(x)
        x = self.global_avg_pool(x)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        ft = x
        x = self.classifier(x)
        return x, ft, first_block


class BranchyNetClassifier(nn.Module):
    def __init__(self, model):
        super(BranchyNetClassifier, self).__init__()

        self.num_classes = 1000 #model.num_classes
        self.model = OFABranchyNetModel( model )
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)

        dropout_rate = 0.2
        feature_dim = 24
        mid_dim = 64
        last_channel = 64
        # final expand layer
        self.final_expand_layer = ConvLayer(
            feature_dim, mid_dim, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        # feature mix layer
        self.feature_mix_layer = ConvLayer(
            mid_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
	)
        # classifier
        self.classifier = LinearLayer(last_channel, self.num_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        x, ft, first_block = self.model(x)
        x = first_block

        x = self.final_expand_layer(x)
        x = self.global_avg_pool(x)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        ft = x
        x = self.classifier(x)
        return x, ft


class OFANetModel(nn.Module):
    def __init__(self, model):
        super(OFANetModel, self).__init__()

        self.num_classes = 1000 #model.num_classes
        self.model = model
        self.first_conv = model.first_conv
        self.blocks = model.blocks
        self.final_expand_layer = model.final_expand_layer
        self.global_avg_pool = model.global_avg_pool
        self.feature_mix_layer = model.feature_mix_layer
        self.classifier = model.classifier
        self.skip_len = 2

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

class MCUNetModel(nn.Module):
    def __init__(self, model):
        super(MCUNetModel, self).__init__()

        self.num_classes = 1000 #model.num_classes
        self.model = model
        self.first_conv = model.first_conv
        self.blocks = model.blocks
        self.feature_mix_layer = model.feature_mix_layer
        self.classifier = model.classifier
        self.skip_len = 2

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


def get_model_from_name(model_name, model_type, args, reset_stats=False):
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

    '''

{'wid': None, 'ks': [3, 5, 3, 3, 5, 3, 3, 3, 5, 5, 3, 3, 7, 5, 3, 3, 3, 3, 7, 5], 'e': [3, 4, 3, 3, 3, 4, 6, 3, 4, 4, 3, 3, 4, 4, 4, 4, 6, 6, 6, 3], 'd': [2, 3, 2, 4, 4], 'r': [208], 'width': 1.2}
{'wid': None, 'ks': [5, 3, 3, 3, 5, 3, 5, 3, 5, 5, 3, 3, 5, 7, 3, 3, 7, 3, 3, 3], 'e': [4, 3, 3, 3, 4, 4, 4, 3, 4, 6, 3, 3, 6, 6, 6, 6, 6, 6, 6, 3], 'd': [3, 4, 4, 3, 4], 'r': [224], 'width': 1}

    '''

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
        model = create_model(
            model_name,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            #bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint)
        model_cfg = model.default_cfg
        model = TimmModel(model, model_name)
    elif model_type == 'ofa_spec':
        model, image_size = ofa_specialized(net_id=model_name, pretrained=True)
        model_cfg = get_model_config( resolution=image_size )
        #model_cfg['interpolation'] = 'bicubic'
        model = OFANetModel(model)
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
      print('Model type ', model_type, ' not supported!!')
      assert(1==2)
    #print(' model -- ', model_name, ' -- cfg ', model_cfg)
    return model, model_cfg

