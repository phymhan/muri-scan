import math, copy
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, ft_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        # x.shape -> (batch, S*T, ft_dim)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, ft_size, time_len, joint_num, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len  = time_len
        self.domain    = domain

        if domain == "temporal" or domain == "mask_t":
            #temporal positial embedding
            pos_list = list(range(self.joint_num * self.time_len))

        elif domain == "spatial" or domain == "mask_s":
            # spatial positial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, ft_size)
        #position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, ft_size, 2).float() * -(math.log(10000.0) / ft_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).cuda()
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadedAttention(nn.Module):

    def __init__(self, att_heads, latent_dim, input_dim, dp_rate, time_len, joint_num, domain):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.attn       = None                # calculate_att weight
        self.att_heads  = att_heads
        self.latent_dim = latent_dim
        #self.att_ft_dropout = nn.Dropout(p=dp_rate)
        # spatial of tempoal
        self.joint_num  = joint_num
        self.time_len   = time_len
        self.domain     = domain

        self.register_buffer('t_mask', self.get_domain_mask()[0])
        self.register_buffer('s_mask', self.get_domain_mask()[1])

        self.query_map = nn.Sequential(
                            nn.Linear(input_dim, self.latent_dim * self.att_heads),
                            nn.Dropout(dp_rate),
                            )
        self.key_map   = nn.Sequential(
                            nn.Linear(input_dim, self.latent_dim * self.att_heads),
                            nn.Dropout(dp_rate),
                            )
        self.value_map = nn.Sequential(
                            nn.Linear(input_dim, self.latent_dim * self.att_heads),
                            nn.ReLU(),
                            nn.Dropout(dp_rate),
                            )

    def get_domain_mask(self):
        # Sec 3.4
        t_mask = torch.ones(self.time_len * self.joint_num, self.time_len * self.joint_num)
        filted_area = torch.zeros(self.joint_num, self.joint_num)

        for i in range(self.time_len):
            row_begin = i * self.joint_num
            column_begin = row_begin
            row_num = self.joint_num
            column_num = row_num

            #Sec 3.4
            t_mask[row_begin: row_begin + row_num, column_begin: column_begin + column_num] *= filted_area

        I = torch.eye(self.time_len * self.joint_num)
        s_mask = Variable((1 - t_mask)).cuda()
        t_mask = Variable(t_mask + I).cuda()
        return t_mask, s_mask


    def attention(self, query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        # (batch, time, ft_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if self.domain is not None:
            #section 3.4 spatial temporal mask operation
            if self.domain == "temporal":
                scores *= self.t_mask  # set weight to 0 to block gradient
                scores += (1 - self.t_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax
            elif self.domain == "spatial":
                scores *= self.s_mask  # set weight to 0 to block gradient
                scores += (1 - self.s_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax

        # apply weight_mask to bolck information passage between ineer-joint
        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn


    def forward(self, x):
        "Implements Figure 2"
        nbatches = x.size(0) # [batch, t, dim]
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.query_map(x).view(nbatches, -1, self.att_heads, self.latent_dim).transpose(1, 2)
        key = self.key_map(x).view(nbatches, -1, self.att_heads, self.latent_dim).transpose(1, 2)
        value = self.value_map(x).view(nbatches, -1, self.att_heads, self.latent_dim).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value) #[batch, att_heads, T, latent_dim ]

        # 3) "Concat" using a view and apply a final linear
        # ( batch, T, latent_dim * att_heads )
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.latent_dim * self.att_heads)

        return x


class ST_ATT_Layer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, input_size, output_size, att_heads, latent_dim, dp_rate, time_len, joint_num, domain):
        """
        Args:
            input_size : the dimension of input
            output_size: the dimension of output
            att_heads:  # of attention heads
            latent_dim: dim of each att head
            time_len: # of frames
            domain: spatial ot temporal domain to attent
        """
        super(ST_ATT_Layer, self).__init__()

        self.pe = PositionalEncoding(input_size, time_len, joint_num, domain)
        self.attn = MultiHeadedAttention(att_heads, latent_dim, input_size, dp_rate, time_len, joint_num, domain)
        self.ft_map = nn.Sequential(
                        nn.Linear(att_heads * latent_dim, output_size),
                        nn.ReLU(),
                        LayerNorm(output_size),
                        nn.Dropout(dp_rate),
                        )

        self.init_parameters()

    def forward(self, x):
        x = self.pe(x)
        x = self.attn(x)
        x = self.ft_map(x)
        return x

    def init_parameters(self):
        model_list = [self.attn, self.ft_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
