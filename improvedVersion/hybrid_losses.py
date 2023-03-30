'''
Copyright 2023 Anil Kag (https://anilkagak2.github.io)

Hybrid / Abstention Loss functions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Without epsilons, lambdas, and mus
# OSP: Selective Classification via One-Sided Prediction
def osp_loss( logits, targets, num_classes=1000, ):
    eps=1e-7
    tol=1e-8

    y_input = F.one_hot( targets, num_classes )
    y_out   = F.softmax( logits, dim=1 )

    n_pos = torch.sum( y_input, dim=0 ) + 0.1
    n_neg = torch.sum( 1-y_input, dim=0 ) + 0.1

    loss_pos = (1./n_pos) * torch.sum( -y_input * torch.log( y_out + tol ), dim=0 )
    loss_neg = (1./n_neg) * torch.sum( -(1-y_input) * torch.log( 1-y_out + tol ), dim=0 )

    xent = torch.sum(loss_pos + loss_neg) 
    return xent

# DiSK : Distilling Scaffolded Knowledge
def disk_loss( s_logits, t_logits, targets, gate, temp_s=4., temp_t=4., topK=10, num_classes=1000, ):
    y_one_hot = F.one_hot( targets, num_classes )

    topk, indices = torch.topk( t_logits, topK ) 
    one_hot = torch.sum( F.one_hot( indices, num_classes=num_classes ), dim=1 )
    one_hot = torch.max( one_hot, y_one_hot )

    z = (s_logits / temp_s) 
    z = F.softmax( z, dim=1 )
    z = z + gate.view(-1, 1)  * one_hot
    N = z.size(0)

    min_vals, _ = torch.min(t_logits, 1, keepdim=True)
    ty = (t_logits / temp_t) * one_hot + min_vals * (1-one_hot)  
    ty = F.softmax( ty, dim=1 ) 
    soft_teacher = ty * one_hot

    disk_loss = temp_s * temp_t * torch.sum( - soft_teacher * torch.log(z) ) / N
    return disk_loss


def hybrid_oracle():
    pass

def hybrid_router_loss():
    pass

def hybrid_student_loss():
    pass

def hybrid_teacher_loss():
    pass
