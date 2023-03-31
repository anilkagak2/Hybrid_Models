import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from timm import utils

def pretty_print_cov_acc( pd_data ):
    df = pd.DataFrame( pd_data, columns=['type', 'global-model', 'base-model', 
                    'global-flops', 'global-acc', 'base-flops', 
                    'base-acc', 'base-cov', 'base@cov', 'global@cov', 
                    'hybrid-acc', 'hybrid-flops'] )
    print(df.to_markdown()) 

def add_standalone_stats( pd_data, global_model_stats, global_model_name ):
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

def eval_hybrid_cov_acc( args, all_tensors, pd_data, model_stats, global_model_stats, hybrid_model_stats  ):
    all_s_pred, all_t_pred, all_y_true, all_s_entropy, all_hybrid_gate, all_disk_gate = all_tensors

    s_acc = torch.mean( (all_s_pred == all_y_true) * 1. ) * 100.
    t_acc = torch.mean( (all_t_pred == all_y_true) * 1. ) * 100.

    model_stats[args.model + 'valid_acc1'] = s_acc.item()
    global_model_stats[args.global_model + 'valid_acc1'] = t_acc.item()

    add_standalone_stats( pd_data, model_stats, args.model )
    add_standalone_stats( pd_data, global_model_stats, args.global_model )

    pretty_print_cov_acc( pd_data )
    
    for scheme in ['agreement', 'margin', 'margin-upper']:
        add_hybrid_stats_in_table( pd_data, args, all_tensors, model_stats, global_model_stats, hybrid_model_stats, scheme=scheme )    

    cov_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    for scheme in ['entropy', 'gate']:
      for cov in cov_list:
        add_hybrid_stats_in_table( pd_data, args, all_tensors, model_stats, global_model_stats, hybrid_model_stats, scheme=scheme, cov=cov )    

    pretty_print_cov_acc( pd_data )


def get_entropy_thr_at_cov(y_entropy, target_cov=0.9, num=50, low=0, high=5):
    best_thr=-10.0 #-1.84
    _best_cov = 0.99
    for thr in list(np.linspace(low, high, num=num)):
        _cov = torch.mean( (y_entropy <= thr)*1. )
        if _cov >= target_cov and _cov<=_best_cov: 
            _best_cov = _cov
            best_thr = thr
    return best_thr, _best_cov

def get_thr_at_cov(y_gate, target_cov=0.9, num=50, low=-2, high=2):
    best_thr=-10.0 #-1.84
    _best_cov = 0.99
    for thr in list(np.linspace(low, high, num=num)):
        #_cov = np.mean( ((y_gate_vals[:,1]-y_gate_vals[:,0]) >= thr)*1  )
        _cov = torch.mean( (y_gate >= thr)*1.  )
        if _cov >= target_cov and _cov<=_best_cov: 
            _best_cov = _cov
            best_thr = thr
    return best_thr, _best_cov


def add_pd_data(args, pd_data, prefix, global_model_stats, base_model_stats, hybrid_model_stats, scheme_name='Entropy'):
    model_name = args.model
    global_model_name = args.global_model
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

def add_hybrid_stats_in_table( pd_data, args, all_tensors, model_stats, global_model_stats, hybrid_model_stats, scheme='agreement', cov=0.9 ):
    all_s_pred, all_t_pred, all_y_true, all_s_entropy, all_hybrid_gate, all_disk_gate = all_tensors
    model_name = args.model
    global_model_name = args.global_model

    if scheme == 'agreement':
        scheme_name = 'Oracle-Agreement'
        prefix=model_name+'oracle-agreement'
        route= all_s_pred == all_t_pred #base_y_pred==global_y_pred
    elif scheme == 'margin-upper':
        scheme_name = 'Oracle-Margin-upper'
        prefix=model_name+'oracle-margin-upper'
        route = all_s_pred == all_y_true #base_y_pred==base_y_true 
    elif scheme == 'margin':
        scheme_name = 'Oracle-Margin'
        prefix=model_name+'oracle-margin'
        #route = np.logical_or(base_y_pred==base_y_true, (global_y_pred!=base_y_true) ) 
        route = torch.logical_or( all_t_pred != all_y_true, all_s_pred == all_y_true )
    elif scheme == 'gate':
        best_thr, _best_cov = get_thr_at_cov(all_hybrid_gate, target_cov=cov, num=2000, low=-4, high=4)
        #print('[Gating] Best cov = ', _best_cov, ' at thr=', best_thr)
        thr=best_thr #-1.84
        scheme_name = 'Gating-' + '{:.2f}'.format(cov)
        #prefix=model_name+'-gating-vals-'+ str(thr)  + '{:.2f}'.format(cov)
        prefix=model_name+'-gating-vals-'+ '{:.2f}'.format(cov)
        #route = ((y_gate_vals[:,1]-y_gate_vals[:,0]) >= thr)*1 
        route = (all_hybrid_gate >= thr) 
    elif scheme == 'entropy':
        scheme_name = 'Entropy-' + '{:.2f}'.format(cov)
        prefix=model_name+'entropy-' + '{:.2f}'.format(cov)
        best_thr, _best_cov = get_entropy_thr_at_cov(all_s_entropy, target_cov=cov, num=500, low=0, high=5)
        #print('[Entropy] Best cov = ', _best_cov, ' at thr=', best_thr)
        found_th = best_thr
        route=all_s_entropy <= found_th  #base_y_entropy<=found_th
    else:
        raise NotImplementedError 

    add_hybrid_stats( args, all_tensors, model_stats, global_model_stats, hybrid_model_stats, route, torch.logical_not(route), prefix=prefix, )

    add_pd_data( args, pd_data, prefix, global_model_stats, model_stats, hybrid_model_stats, scheme_name=scheme_name, )

def add_hybrid_stats( args, all_tensors, model_stats, global_model_stats, hybrid_model_stats, gate_base, gate_global,  prefix='entropy', ): 
    all_s_pred, all_t_pred, all_y_true, all_s_entropy, all_hybrid_gate, all_disk_gate = all_tensors
    base_name = args.model
    global_name = args.global_model

    hybrid_pred = ( all_s_pred * gate_base ) + ( all_t_pred * gate_global )  
    hybrid_cov = torch.sum( 1.*gate_base ) / len(all_y_true)
    hybrid_acc = torch.sum( 1.*(hybrid_pred == all_y_true) ) / len(all_y_true)
    hybrid_flops = model_stats[base_name+'flop'] + (1. - hybrid_cov) * global_model_stats[global_name+'flop']

    global_abstained_acc = torch.sum( 1.* (all_t_pred == all_y_true) * gate_global ) / torch.sum( 1.*gate_global )
    base_predicted_acc = torch.sum( 1.* (all_s_pred == all_y_true) * gate_base ) / torch.sum( 1.*gate_base )

    hybrid_model_stats[prefix + '_hybrid-valid_acc'] = hybrid_acc*100
    hybrid_model_stats[prefix + '_hybrid-cov'] = hybrid_cov*100
    hybrid_model_stats[prefix + '_hybrid-flop'] = hybrid_flops
    hybrid_model_stats[prefix + '_hybrid-global_abs_acc'] = global_abstained_acc*100
    hybrid_model_stats[prefix + '_hybrid-base_pred_acc'] = base_predicted_acc*100

