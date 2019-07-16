from .transformer_spatial_tempor import *
import torchvision
import torch.nn as nn
import torch
from .gcn_layer import GraphConvolution
import numpy as np
import torch.nn.functional as F

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.permute(self.shape)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, ft_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        #[batch, time, ft_dim)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# class My_BN(nn.Module):
#     "Construct a layernorm module (See citation for details)."
#     def __init__(self, ft_dim):
#         super(My_BN, self).__init__()
#         self.bn = nn.BatchNorm1d(ft_dim)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # [batch, 3, 22*8]
#         x = self.bn(x)
#         x = x.permute(0, 2, 1)
#         return x




# class MHA_Pool(nn.Module):
#     def __init__(self, h_num, h_dim, input_dim, dp_rate):
#         "Take in model size and number of heads."
#         super(MHA_Pool, self).__init__()
#         #assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.h_dim = h_dim # head dimension
#         self.h_num = h_num #head num
#         self.attn = None #calculate_att weight
#         #self.att_ft_dropout = nn.Dropout(p=dp_rate)
#         self.key_map = nn.Sequential(
#                             nn.Linear(input_dim, self.h_dim * self.h_num),
#                             nn.Dropout(dp_rate),
#                             )
#
#
#         self.query_map = nn.Sequential(
#                             nn.Linear(input_dim, self.h_dim * self.h_num),
#                             nn.Dropout(dp_rate),
#                             )
#
#
#         self.value_map = nn.Sequential(
#                             nn.Linear(input_dim, self.h_dim * self.h_num),
#                             nn.Dropout(dp_rate),
#                                      )
#
#
#     def attention(self,query, key, value):
#         "Compute 'Scaled Dot Product Attention'"
#         # [batch, time, ft_dim)
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(d_k)
#         #print("scores.shape:",scores.shape)
#         p_attn = F.softmax(scores, dim=-1)
#
#         # add spatial_mask ---> only get information from different time_step
#         return torch.matmul(p_attn, value), p_attn
#
#     def forward(self, x):
#         "Implements Figure 2"
#         avg_pool_x = x.sum(1,keepdim = True) #[batch,1, input_dim]
#         nbatches = x.size(0) # [batch, t, dim]
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#
#         #att = Query * key
#         query = self.query_map(avg_pool_x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)# [batch, head_num,1, dim]
#         #print("query.shape:",query.shape)
#         key = self.key_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)# [batch, head_num,t, dim]
#         #print("key.shape:", key.shape)
#         value = self.value_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
#         #print("value.shape:", value.shape)
#
#         #print("query.shape:",query.shape)
#
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = self.attention(query, key, value) #[batch, h_num, T, h_dim ]
#
#             # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous() \
#             .view(nbatches, -1, self.h_dim * self.h_num)#[batch, 1, h_dim * h_num ]
#
#         #print("att_ft.shape:", x.shape)
#         #x = self.att_ft_dropout(x)
#         #x = torch.cat((avg_pool_x,x), -1)
#         x = x.squeeze()
#         return x



class Trans_branch(nn.Module):
    def __init__(self, input_size, use_graph,dp_rate):
        super(Trans_branch, self).__init__()

        self.time_len = 8
        h_dim = 32
        h_num= 8
        self.dp_rate = dp_rate


        self.input_map = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(dp_rate),
            #nn.Linear(64, 64),
            #nn.ReLU(),
            #LayerNorm(64),
            #nn.Dropout(dp_rate),
        )
        #self.initialize_weights(self.input_map)

        use_pe = False
        self.trans_net = nn.ModuleList(
            # N, input_size, h_num, h_dim, output_size, dp_rate, time_len, use_graph, use_pe
            [Trans_Encoder(N=1, input_size=128, output_size=128, h_num=h_num, h_dim = h_dim, dp_rate=dp_rate, time_len=self.time_len,use_graph=use_graph,use_pe = use_pe,domain="spatial"),
             #Trans_Encoder(N=1, input_size=64, output_size=64, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=self.time_len, use_graph=use_graph, use_pe=True, domain="temporal"),

            Trans_Encoder(N=1, input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=self.time_len,use_graph=use_graph, use_pe= use_pe, domain="temporal"),]
             #Trans_Encoder(N=1, input_size=256, output_size=256, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len=self.time_len,use_graph=use_graph, use_pe=True, domain="temporal")]



            #Trans_Encoder(N=1, input_size=128, output_size=128, h_num=1, h_dim=128, dp_rate=dp_rate, time_len=8, use_graph=use_graph, use_pe=True, domain="att_pool"),
        )

        # self.mask_net = nn.ModuleList(
        #     # N, input_size, h_num, h_dim, output_size, dp_rate, time_len, use_graph, use_pe
        #     [Trans_Encoder(N=1, input_size=64, output_size=128, h_num=8, h_dim = 32, dp_rate=dp_rate, time_len=self.time_len,use_graph=use_graph,use_pe = True,domain="mask"),
        #
        #     Trans_Encoder(N=1, input_size=128, output_size=256, h_num=8, h_dim=32, dp_rate=dp_rate, time_len=self.time_len,use_graph=use_graph, use_pe=True, domain="mask")
        #      ]
        # )


    def forward(self, x):

        x = self.input_map(x)
        for trans_layer in self.trans_net:
            #mask = mask_layer(x)
            x = trans_layer(x)
            #x = mask * x + x
        x = x.sum(1) / x.shape[1]

       #x = self.att_pool(x)

        return x

    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




class Trans_only(nn.Module):
    def __init__(self, num_classes, use_graph, dp_rate):
        super(Trans_only, self).__init__()
        self.feature_size = 3

        self.T = 8
        self.trans_ft_dim = 128

        self.trans_encoder = Trans_branch(self.feature_size, use_graph, dp_rate)

        #self.input_bn = nn.BatchNorm1d(self.feature_size)

        self.cls = nn.Linear(self.trans_ft_dim, num_classes)



    def forward(self, x):
        # x [batch, T, joint_num, ft_dim]
        #batch_size = x.shape[0]
        seq_num = x.shape[1]
        for t in range(seq_num):
            x_t = x[:, t, :, :] #[batch,22,3]
            if t == 0:
                gcn_input = x_t
            else:
                gcn_input = torch.cat((gcn_input,x_t), 1)


        # gcn_input = gcn_input.permute(0, 2, 1)  # [batch, 3, 22*8]
        # gcn_input = self.input_bn(gcn_input)
        # gcn_input = gcn_input.permute(0, 2, 1)

        #gcn_input = self.input_norm(gcn_input)
        x = self.trans_encoder(gcn_input)
        x = self.cls(x)


        return x

    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)