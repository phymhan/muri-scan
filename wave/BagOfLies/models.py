import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import utils
import numpy as np
<<<<<<< HEAD
import functools


class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
=======
>>>>>>> 33399534affccf16ee9ff03c070018ed48695c24


def conv_block(in_ch, out_ch, L, activation='relu'):
    
<<<<<<< HEAD
    return 


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.empty((1, channel_size, 1)))
        self.beta = nn.Parameter(torch.empty((1, channel_size, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2,
                                                keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1,
                                            keepdim=True).mean(dim=2,
                                                               keepdim=True)
        gLN_y = (self.gamma * (y - mean) /
                 torch.pow(var + 10e-8, 0.5) + self.beta)
        return gLN_y


class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class NewModel(nn.Module):

    class Block( nn.Module):
        def __init__( self, B, P, H, D):
            super( NewModel.Block, self).__init__()

            self.m = nn.ModuleList( [
                nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1),
                nn.PReLU(),
                GlobalLayerNorm(H),
                nn.Conv1d(in_channels=H, out_channels=H, kernel_size=P,
                          padding=(D * (P - 1)) // 2, dilation=D, groups=H),
                nn.PReLU(),
                GlobalLayerNorm(H),
                nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l(y)
            return x + y


    def __init__(self, opt):
        super(NewModel, self).__init__()
        if opt.dropout > 0:
            drop_layer = functools.partial(nn.Dropout, p=opt.dropout)
        else:
            drop_layer = Identity
        self.L         = opt.kernel_size
        self.input_dim = opt.feature_dim
        self.embed_dim = opt.embed_dim
        self.activations = nn.ModuleDict([
                ['relu', nn.ReLU()],
                ['lrelu', nn.LeakyReLU(0.3)],
                ['prelu', nn.PReLU()],
                ['elu', nn.ELU(alpha=1.0)],
                ['softplus', nn.Softplus()],
                ['sigmoid', nn.Sigmoid()]
        ])
        self.conv_block1 = nn.Sequential(
                nn.Conv1d(in_channels=self.input_dim, out_channels=self.embed_dim, 
                        kernel_size=self.L, stride=self.L//2, padding=self.L//2, bias=False),
                self.activations[opt.activation],
                drop_layer(),
                nn.BatchNorm1d(self.embed_dim)
            )
        # dense layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1),
            drop_layer()
        )

        self.sm = nn.ModuleList([
            NewModel.Block(B=self.embed_dim, P=3, H=64, D=2**d) for _ in range(opt.num_blocks) for d in range(1)
        ])

        self.avg_pool  = nn.AdaptiveAvgPool1d(1)
        self.fc_target = nn.Linear(self.embed_dim, opt.num_classes)
        # self.fc_target = nn.Linear(128, opt.num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv2(x)
        for l in self.sm:
            x = l(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_target(x)
        return x


class TCN(nn.Module):

    class Block( nn.Module):
        def __init__( self, B, P, D):
            super( TCN.Block, self).__init__()
=======
    
    return 


class TSN(nn.Module):

    class Block( nn.Module):
        def __init__( self, B, P, D):
            super( TSN.Block, self).__init__()
>>>>>>> 33399534affccf16ee9ff03c070018ed48695c24

            self.m = nn.ModuleList( [
              nn.Conv1d(in_channels=B, out_channels=B, kernel_size=P,
                        padding=(D*(P-1))//2, dilation=D),
              nn.ReLU(),
              nn.BatchNorm1d(B),
            ])

        def forward(self, x):
            y = x.clone()
            for l in self.m:
                y = l( y)
            return x+y


    def __init__(self, opt):
<<<<<<< HEAD
        super(TCN, self).__init__()
        if opt.dropout > 0:
            drop_layer = functools.partial(nn.Dropout, p=opt.dropout)
        else:
            drop_layer = Identity
=======
        super(TSN, self).__init__()
>>>>>>> 33399534affccf16ee9ff03c070018ed48695c24
        self.L         = opt.kernel_size
        self.input_dim = opt.feature_dim
        self.embed_dim = opt.embed_dim
        self.activations = nn.ModuleDict([
                ['relu', nn.ReLU()],
                ['lrelu', nn.LeakyReLU(0.3)],
                ['prelu', nn.PReLU()],
                ['elu', nn.ELU(alpha=1.0)],
                ['softplus', nn.Softplus()],
                ['sigmoid', nn.Sigmoid()]
        ])
        self.conv_block1 = nn.Sequential(
                nn.Conv1d(in_channels=self.input_dim, out_channels=self.embed_dim, 
                        kernel_size=self.L, stride=self.L//2, padding=self.L//2, bias=False),
                self.activations[opt.activation],
<<<<<<< HEAD
                drop_layer(),
                nn.BatchNorm1d(self.embed_dim)
            )
        # dense layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1),
            drop_layer()
        )

        self.sm = nn.ModuleList([
            TCN.Block(B=self.embed_dim, P=3, D=2**d) for _ in range(opt.num_blocks) for d in range(1)
=======
                nn.BatchNorm1d(self.embed_dim)
            )
        # dense layer
        self.conv2 = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1)

        self.sm = nn.ModuleList([
            TSN.Block(B=self.embed_dim, P=3, D=2**d) for _ in range(opt.num_blocks) for d in range(1)
>>>>>>> 33399534affccf16ee9ff03c070018ed48695c24
        ])

        self.avg_pool  = nn.AdaptiveAvgPool1d(1)
        self.fc_target = nn.Linear(self.embed_dim, opt.num_classes)
        # self.fc_target = nn.Linear(128, opt.num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv2(x)
        # for l in self.sm:
        #     x = l(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_target(x)
        return x


<<<<<<< HEAD
if __name__ == "__main__":
    opt = utils.Options().get_options()
    model = TCN(opt)
=======

if __name__ == "__main__":
    opt = utils.Options().get_options()
    model = TSN(opt)
>>>>>>> 33399534affccf16ee9ff03c070018ed48695c24
    summary(model.cuda(), input_size=(opt.feature_dim, opt.num_frames))
