

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

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
        self.K = 20
        n_ft = self.K * self.K + 1 + num_ft + 1000 #1000 + 1 + 100 + 576 #1024
        n_ft2 = 256
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
            nn.Linear( 3 * n_ft2, 64, bias=True),
            #nn.Linear( n_ft2, 64, bias=True),
            #nn.Linear( 2*n_ft2, 64, bias=True),
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
        x2 = self.R2( ft )
        #x3 = self.R3( softmax )
        x3 = self.R3( logits )

        gate = self.clf( torch.cat([x1, x2, x3], dim=1) )
        #gate = self.clf( torch.cat([x1, x3], dim=1) )
        #gate = self.clf(x1)
        #gate = self.routing(x)
        #print('gate = ', gate.size())
        #assert(1==2)
        #return ft, logits, gate
        return gate

class RoutingNetworkTop20(nn.Module):
    def __init__(self):
        super(RoutingNetworkTop20, self).__init__()
        self.flatten = nn.Flatten()
        self.K = 20
        n_ft = 1 + self.K*self.K #1000 + 1 + 100 + 576 #1024
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
        C = self.K
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



class RoutingNetworkTop20WithFt(nn.Module):
    def __init__(self, num_ft):
        super(RoutingNetworkTop20WithFt, self).__init__()
        self.flatten = nn.Flatten()
        self.K = 20
        n_ft = 1 + self.K*self.K + num_ft #1000 + 1 + 100 + 576 #1024
        print('Top10x10 -- Total routing ft = ', n_ft)
        self.routing = nn.Sequential(
            #nn.BatchNorm1d( n_ft ),
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
        C = self.K
        topk = torch.topk( softmax, C, dim=1 )[0]
        y_margin = topk.view(B, C, -1) - topk.view(B, -1, C)
        y_margin = y_margin.view(B, -1)

        #x = torch.cat([entropy, y_margin], dim=1)
        x = torch.cat([entropy, y_margin, ft], dim=1)
        #x = torch.cat([entropy, softmax, y_margin, ft], dim=1)
        #x = torch.cat([entropy, logits, softmax, logs], dim=1)
        #print('x = ', x.size())

        x = self.flatten(x)
        gate = self.routing(x)
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


#def get_routing_model( base_checkpoint=None, device='cuda', routing_name='', base_model_cfg=None ):
def get_routing_model( routing_name='', base_model_cfg=None ):
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
    #routingNet = torch.nn.DataParallel(routingNet)
    #routingNet = routingNet.to(device)

    #if base_checkpoint is not None:
    #    routingNet.load_state_dict(base_checkpoint["routing_state_dict"])
    return routingNet

