
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class DiSK_Router(nn.Module):
    def __init__(self, n_labels=1000, num_features=-1):
        print('---------------------------------- DiSK Router (teacher_ft) --')
        super(ImageNet_Routing_B2, self).__init__()

        self.n_labels = n_labels
        n_ft = 64

        self.routing = nn.Sequential(
                nn.Linear(num_features, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
                nn.Linear(n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, 1), # bias=True),
                nn.Sigmoid(),
            )

    def forward(self, t_logits, t_ft ):
        gate = self.routing( t_ft )
        return gate  


class Hybrid_Router(nn.Module):
    def __init__(self, n_labels=1000, num_features=-1):
        print('---------------------------------- Hybrid Router (student_ft) --')
        super(ImageNet_Routing_B2, self).__init__()

        self.K = 10
        self.n_labels = n_labels
        n_ft = 64

        self.transform_ft = nn.Sequential(
                nn.Linear(num_features, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
            )

        logit_ft_num = 1 + self.K**2
        self.routing = nn.Sequential(
                nn.Linear(logit_ft_num + n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
                nn.Linear(n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, 1), # bias=True),
                nn.Sigmoid(),
            )

    def process_logits(self, logits):
        softmax = F.softmax( logits, dim=1 )
        entropy = Categorical( probs=softmax ).entropy().unsqueeze(1)

        B = logits.shape[0]
        C = self.K
        topk = torch.topk( softmax, C, dim=1 )[0]
        y_margin = topk.view(B, C, -1) - topk.view(B, -1, C)
        y_margin = y_margin.view(B, -1)

        x = torch.cat([entropy, y_margin], dim=1)
        return x 

    def forward(self, s_logits, s_ft ):
        s_ft = self.transform_ft( s_ft )
        s_logits = self.process_logits( s_logits )

        x = torch.cat([ s_logits, s_ft ], dim=1)
        gate = self.routing( x )
        return gate  


def get_gating_model( routing_name = 'DiSK_Router', n_labels=1000, t_num_ft=-1 ):
    if routing_name=='DiSK_Router':
        routingNet = DiSK_Router( n_labels = n_labels, num_features=t_num_ft )
    elif routing_name=='DiSK_Router':
        routingNet = Hybrid_Router( n_labels = n_labels, num_features=t_num_ft )
    else:
        assert(1==2)
    return routingNet
