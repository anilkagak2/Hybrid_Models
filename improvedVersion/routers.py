
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def get_gating_model( routing_name = 'DiSK_Router', n_labels=1000, t_num_ft=-1 ):
    if routing_name=='DiSK_Router':
        routingNet = DiSK_Router( n_labels = n_labels, num_features=t_num_ft )
    else:
        assert(1==2)
    return routingNet
