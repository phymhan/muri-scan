import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import utils
import numpy as np
import functools


class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def conv_block(in_ch, out_ch, L, activation='relu'):
    
    
    return 


class TCN(nn.Module):

    class Block( nn.Module):
        def __init__( self, B, P, D):
            super( TCN.Block, self).__init__()

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
        super(TCN, self).__init__()
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
            TCN.Block(B=self.embed_dim, P=3, D=2**d) for _ in range(opt.num_blocks) for d in range(1)
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


if __name__ == "__main__":
    opt = utils.Options().get_options()
    model = TCN(opt)
    summary(model.cuda(), input_size=(opt.feature_dim, opt.num_frames))
