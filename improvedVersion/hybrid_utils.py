import numpy as np
import torch
from torch.distributions import Categorical
from timm import utils

from hybrid_models import execute_network

def add_hybrid_stats( base_y_pred, global_y_pred, base_y_true, gate_base, gate_global, 
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

def get_entropy_thr_at_cov(y_entropy, target_cov=0.9, num=50, low=0, high=5):
    best_thr=-10.0 #-1.84
    _best_cov = 0.99
    for thr in list(np.linspace(low, high, num=num)):
        _cov = np.mean( (y_entropy <= thr)*1 )
        if _cov >= target_cov and _cov<=_best_cov: 
            _best_cov = _cov
            best_thr = thr
    return best_thr, _best_cov

def get_thr_at_cov(y_gate, target_cov=0.9, num=50, low=-2, high=2):
    best_thr=-10.0 #-1.84
    _best_cov = 0.99
    for thr in list(np.linspace(low, high, num=num)):
        #_cov = np.mean( ((y_gate_vals[:,1]-y_gate_vals[:,0]) >= thr)*1  )
        _cov = np.mean( (y_gate >= thr)*1  )
        if _cov >= target_cov and _cov<=_best_cov: 
            _best_cov = _cov
            best_thr = thr
    return best_thr, _best_cov


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

def add_hybrid_stats_in_table( pd_data, base_y_pred, global_y_pred, base_y_true,  base_y_entropy, y_gate,
          model_name, global_model_name, base_model_stats, global_model_stats, hybrid_model_stats, scheme='agreement', cov=0.9 ):
    if scheme == 'agreement':
        scheme_name = 'Oracle-Agreement'
        prefix=model_name+'oracle-agreement'
        route=base_y_pred==global_y_pred
    elif scheme == 'margin-upper':
        scheme_name = 'Oracle-Margin-upper'
        prefix=model_name+'oracle-margin-upper'
        route = base_y_pred==base_y_true 
    elif scheme == 'margin':
        scheme_name = 'Oracle-Margin'
        prefix=model_name+'oracle-margin'
        route = np.logical_or(base_y_pred==base_y_true, (global_y_pred!=base_y_true) ) 
    elif scheme == 'gate':
        best_thr, _best_cov = get_thr_at_cov(y_gate, target_cov=cov, num=2000, low=-4, high=4)
        #print('[Gating] Best cov = ', _best_cov, ' at thr=', best_thr)
        thr=best_thr #-1.84
        scheme_name = 'Gating-' + '{:.2f}'.format(cov)
        #prefix=model_name+'-gating-vals-'+ str(thr)  + '{:.2f}'.format(cov)
        prefix=model_name+'-gating-vals-'+ '{:.2f}'.format(cov)
        #route = ((y_gate_vals[:,1]-y_gate_vals[:,0]) >= thr)*1 
        route = (y_gate >= thr)*1 
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


def evaluate_routing_model( args, model, global_model, disk_router, hybrid_router, 
                            xloader, batch_size, device ):
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    Tlosses_m = utils.AverageMeter()
    Ttop1_m = utils.AverageMeter()
    Ttop5_m = utils.AverageMeter()

    B = len(xloader.dataset) // batch_size
    N = len(xloader.dataset) #len(loader) * args.batch_size
    s_pred = np.zeros( (N,) )
    t_pred = np.zeros( (N,) )
    y_true = np.zeros( (N,) )
    y_entropy = np.zeros( (N,) )
    y_hybrid_gate = np.zeros( (N,) )
    y_disk_gate = np.zeros( (N,) )

    model.eval()
    global_model.eval()
    disk_router.eval()
    hybrid_router.eval()

    with torch.no_grad():
      for i, (input, g_input, target) in enumerate(xloader):
        if not args.prefetcher:
            input = input.to(device)
            g_input = g_input.to(device)
            target = target.to(device)

        s_logits, t_logits, hybrid_gate, disk_gate = execute_network( 
                   model, global_model, disk_router, hybrid_router, 
                   input, g_input, args, train=False )

        _start = i*batch_size
        _end   = (i+1)*batch_size
        if _end > N: _end=N

        softmax = F.softmax( s_logits, dim=1 )
        entropy = Categorical( probs=softmax ).entropy()
        y_entropy[ _start:_end ] = entropy.detach().cpu().numpy()

        s_pred[ _start:_end ] = torch.argmax( s_logits, dim=1 ).detach().cpu().numpy()
        t_pred[ _start:_end ] = torch.argmax( t_logits, dim=1 ).detach().cpu().numpy()
        y_true[ _start:_end ] = target.detach().cpu().numpy()

        y_hybrid_gate[ _start:_end ] = hybrid_gate.detach().cpu().numpy()
        y_disk_gate[ _start:_end ] = disk_gate.detach().cpu().numpy()

        loss = F.cross_entropy(s_logits, target)
        Tloss = F.cross_entropy(t_logits, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        Tacc1, Tacc5 = utils.accuracy(g_output, target, topk=(1, 5))

        losses_m.update(loss.item(), input.size(0))
        top1_m.update(acc1.item(), s_logits.size(0))
        top5_m.update(acc5.item(), s_logits.size(0))

        Tlosses_m.update(Tloss.item(), input.size(0))
        Ttop1_m.update(Tacc1.item(), t_logits.size(0))
        Ttop5_m.update(Tacc5.item(), t_logits.size(0))
 
    return top1_m.avg, Ttop1_m.avg, s_pred, t_pred, y_true, y_entropy, y_hybrid_gate, y_disk_gate
