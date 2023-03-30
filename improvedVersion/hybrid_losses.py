'''
Copyright 2023 Anil Kag (https://anilkagak2.github.io)

Hybrid / Abstention Loss functions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Without epsilons, lambdas, and mus
# OSP: Selective Classification via One-Sided Prediction
def osp_loss( logits, targets, args, y_one_hot, weights ):
    eps=1e-7
    tol=1e-8

    y_input = y_one_hot #F.one_hot( targets, args.num_classes )
    y_out   = F.softmax( logits, dim=1 )

    n_pos = torch.sum( weights * (y_input), dim=0 ) + 0.1
    n_neg = torch.sum( weights * (1-y_input), dim=0 ) + 0.1

    loss_pos = (1./n_pos) * torch.sum( - weights * y_input * torch.log( y_out + tol ), dim=0 )
    loss_neg = (1./n_neg) * torch.sum( - weights * (1-y_input) * torch.log( 1-y_out + tol ), dim=0 )

    xent = torch.sum(loss_pos + loss_neg) 
    return xent

# DiSK : Distilling Scaffolded Knowledge
def disk_loss( s_logits, t_logits, targets, gate, args, y_one_hot ):
    topk, indices = torch.topk( t_logits, args.topK ) 
    one_hot = torch.sum( F.one_hot( indices, num_classes=args.num_classes ), dim=1 )
    one_hot = torch.max( one_hot, y_one_hot )

    z = (s_logits / args.temp_s) 
    z = F.softmax( z, dim=1 )
    z = z + gate.view(-1, 1)  * one_hot
    N = z.size(0)

    min_vals, _ = torch.min(t_logits, 1, keepdim=True)
    ty = (t_logits / args.temp_t) * one_hot + min_vals * (1-one_hot)  
    ty = F.softmax( ty, dim=1 ) 
    soft_teacher = ty * one_hot

    disk_loss = args.temp_s * args.temp_t * torch.sum( - soft_teacher * torch.log(z) ) / N
    return disk_loss

# Disk budget
def disk_budget_loss( args, gate, s_logits, t_pred, n_incorrect ):
    b_correct = F.cross_entropy( s_logits, t_pred, reduction='none' )
    b_correct = torch.clamp( b_correct, max=args.max_ce )  
    b_correct = b_correct.view(-1, 1)

    budget_loss = args.lmbda * F.relu( (1/n_incorrect) * torch.sum( gate * b_correct ) - args.budget_g )
    return budget_loss

# Label smoothing cross-entropy loss
def label_smoothing_ce_loss( logits, targets, args, weights ):
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = (1. - args.smoothing) * nll_loss + args.smoothing * smooth_loss
 
    loss = weights * loss
    return loss.mean()

# oracle defn.
def hybrid_oracle( s_logits, t_logits, targets ):
    t_pred = torch.argmax( t_logits, dim=1 )
    s_pred = torch.argmax( s_logits, dim=1 )
    oracle = torch.logical_or( t_pred != targets, s_pred == targets )
    n_incorrect = torch.sum( (targets != s_pred) * 1. )
    return oracle, s_pred, t_pred, n_incorrect

def hybrid_router_loss( router, oracle, args ):
    clf_loss = F.binary_cross_entropy( router, oracle )
    cov_loss = F.relu( torch.mean( router ) - args.cov ) 
    return clf_loss + cov_loss

def hybrid_loss( s_logits, t_logits, disk_gate, hybrid_gate, targets, args ):
    y_one_hot = F.one_hot( targets, args.num_classes )
    oracle, s_pred, t_pred, n_incorrect = hybrid_oracle( s_logits, t_logits, targets )

    s_weights = 1. + oracle
    t_weights = 2. - oracle

    #s_weights = 1. + hybrid_gate
    #t_weights = 2. - hybrid_gate

    # Student loss .
    ce_loss = label_smoothing_ce_loss( s_logits, targets, args, s_weights )
    abstention_loss = osp_loss( s_logits, targets, args, y_one_hot, s_weights )

    # Teacher loss .
    #t_ce_loss = label_smoothing_ce_loss( s_logits, targets, args, t_weights )
    #t_abstention_loss = osp_loss( s_logits, targets, args, y_one_hot, t_weights )

    # DiSK losses
    d_loss = disk_loss( s_logits, t_logits, targets, disk_gate, args, y_one_hot )
    d_budget = disk_budget_loss( args, disk_gate, s_logits, t_pred, n_incorrect )

    # Hybrid Router loss
    h_router_loss = hybrid_router_loss( hybrid_gate, oracle, args )

    loss = h_router_loss \
         + d_budget + d_loss \
         + ce_loss + abstention_loss 

    return loss

