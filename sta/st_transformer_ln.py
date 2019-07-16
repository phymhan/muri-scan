import torchvision
import torch.nn as nn
import torch
from .gcn_layer import GraphConvolution
import numpy as np
from torch.autograd import Variable
import math, copy
import torch.nn.functional as F
import numpy as np

use_domain_mask = True
print("use_domain_mask:", use_domain_mask)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])







class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dp_rate, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = 22
        self.time_len = time_len

        self.domain = domain

        #self.dropout = nn.Dropout(dp_rate)


        if domain == "temporal" or domain == "mask_t":
            pos_list = list(range(self.joint_num * self.time_len))
            trainable_pe = False
            # pos_list = []
            # for t in range(self.time_len):
            #     for j_id in range(self.joint_num):
            #         pos_list.append(t)

        elif domain == "spatial" or domain == "mask_s":
            trainable_pe = False
            #pos_list = list(range(self.joint_num * self.time_len))
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        elif domain == "mask":
            trainable_pe = False
            pos_list = list(range(self.joint_num * self.time_len))

        else:
            raise ValueError("No {} domain".format(domain))

        print("domain:", domain, "pos_list:", pos_list)
        print("trainable_pe:",trainable_pe)

        if trainable_pe:
            self.pe = nn.Parameter(torch.randn(1,self.time_len * self.joint_num, d_model))
            print("pe.requires_grad:",self.pe.requires_grad)
        else:
            position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

            # Compute the positional encodings once in log space.
            pe = torch.zeros(self.time_len * self.joint_num, d_model)
            #position = torch.arange(0, max_len).unsqueeze(1)

            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).cuda()
            self.register_buffer('pe', pe)
            print("pe.requires_grad:",self.pe.requires_grad)

    def forward(self, x):
        # if self.domain == "temporal" and self.pe.shape == (1,176,32):
        #     print(self.pe[0][0])
        x = x + self.pe[:, :x.size(1)]
        #x =  self.dropout(x)
        return x


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



class MultiHeadedAttention(nn.Module):
    def __init__(self, h_num, h_dim, input_dim, output_dim, dp_rate, use_graph,domain):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        #assert d_model % h == 0
        # We assume d_v always equals d_k
        self.h_dim = h_dim # head dimension
        self.h_num = h_num #head num
        self.attn = None #calculate_att weight
        #self.att_ft_dropout = nn.Dropout(p=dp_rate)
        self.use_graph = use_graph

        self.register_buffer('s_graph_mask', self.get_spatial_mask())
        self.register_buffer('t_mask', self.get_domain_mask()[0])
        self.register_buffer('s_mask', self.get_domain_mask()[1])

        self.domain = domain# spatial of  tempoal

        self.key_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            #nn.ReLU(),
                            nn.Dropout(dp_rate),
                            )


        self.query_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            #nn.ReLU(),
                            nn.Dropout(dp_rate),
                            )


        self.value_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.ReLU(),
                            nn.Dropout(dp_rate),
                                     )
        if "mask" in self.domain:
            self.ft_map = nn.Sequential(
                            nn.Linear(self.h_dim * self.h_num, output_dim),
                            nn.ReLU(),
                            #LayerNorm(output_dim),
                            #nn.Dropout(dp_rate),
                            nn.Linear(output_dim, output_dim),
                            nn.Sigmoid()
                            )
        else:
            self.ft_map = nn.Sequential(
                            nn.Linear(self.h_dim * self.h_num, output_dim),
                            #nn.ReLU(),
                            #LayerNorm(output_dim),
                            #nn.Dropout(dp_rate),
                            #nn.Linear(output_dim, output_dim),
                            )


    def get_domain_mask(self):
        # use temporal mask
        time_len = 8
        joint_num = 22
        t_mask = torch.ones(time_len * joint_num, time_len * joint_num)
        filted_area = torch.zeros(joint_num, joint_num)

        for i in range(time_len):
            row_begin = i * joint_num
            column_begin = row_begin
            row_num = joint_num
            column_num = row_num

            t_mask[row_begin: row_begin + row_num, column_begin: column_begin + column_num] *= filted_area

        I = torch.eye(time_len * joint_num)

        return Variable(t_mask + I).cuda(), Variable((1 - t_mask)).cuda()

    def get_spatial_mask(self):#use spatial connection
        wrist = [0]
        palm = [1]
        thumb = [2, 3, 4, 5]  # da mu zhi
        fore_finger = [6, 7, 8, 9]  # shi zhi
        middle_finger = [10, 11, 12, 13]  # zhong zhi
        ring_finger = [14, 15, 16, 17]  # wu ming zhi
        little_finger = [18, 19, 20, 21]  # xiaozhi
        hand = [wrist, palm, thumb, fore_finger, middle_finger, ring_finger, little_finger]

        def get_spatial_adj():

            print("Thumb base to wirst..")
            adj = torch.zeros(22, 22)

            # finger connection
            for part_id in range(2, 7):
                part_joint = hand[part_id]
                join_num = len(part_joint)
                for i in range(join_num - 1):
                    id_1 = part_joint[i]
                    id_2 = part_joint[i + 1]
                    adj[id_1][id_2] = 1
                    adj[id_2][id_1] = 1

            # palm to finger
            id_1 = palm[0]
            for part_id in range(3, 7):
                part_joint = hand[part_id]
                id_2 = part_joint[0]
                adj[id_1][id_2] = 1
                adj[id_2][id_1] = 1

            # palm to wrist
            id_1 = palm[0]
            id_2 = wrist[0]
            adj[id_1][id_2] = 1
            adj[id_2][id_1] = 1

            # thumb base to wrist
            id_1 = thumb[0]
            id_2 = wrist[0]
            adj[id_1][id_2] = 1
            adj[id_2][id_1] = 1

            adj += torch.eye(22)
            return adj

        #########
        # set the weights of inner-frame joints as zeros
        #########
        time_len = 8
        joint_num = 22
        mask = torch.ones(time_len * joint_num, time_len * joint_num)
        adj = get_spatial_adj()
        for i in range(time_len):
            row_begin = i * joint_num
            column_begin = row_begin
            row_num = joint_num
            column_num = row_num

            mask[row_begin: row_begin + row_num, column_begin: column_begin + column_num] *= adj

        # print(mask[0])
        # print(mask[1])
        # inf_mask = (1 - mask) * (-9e15)
        print("mask.requires_grad:", mask.requires_grad)
        return Variable(mask).cuda()


        #self.register_buffer('spatial_mask', self.get_spatial_mask())

        #self.weight_mask = self.get_spatial_mask()

    def attention(self,query, key, value, use_graph, domain):
        "Compute 'Scaled Dot Product Attention'"
        # [batch, time, ft_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if use_domain_mask:
            if domain == "temporal" or domain == "mask_t":
                scores *= self.t_mask  # set weight to 0 to block gradient
                scores += (1 - self.t_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax
            elif domain == "spatial" or domain == "mask_s":
                if use_graph:
                    # print("add graph constrain")
                    scores *= self.s_graph_mask  # set weight to 0 to block gradient
                    scores += (1 - self.s_graph_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax

                else:
                    scores *= self.s_mask  # set weight to 0 to block gradient
                    scores += (1 - self.s_mask) * (-9e15)  # set weight to -inf to remove effect in Softmax

        # apply weight_mask to bolck information passage between ineer-joint

        p_attn = F.softmax(scores, dim=-1)

        # if domain == "spatial":
        #     print(p_attn[0][0][1])

        # if use_graph:
        #     print(p_attn[0][0][23])

        # #print(scores.shape)
        # result_list = []
        # time = 0
        # #print(p_attn.shape)
        # print("*"*40)
        # tag_data = -1
        # contrib = p_attn[tag_data][1].sum(0)
        # max_idx =  contrib.argmax().item()
        # max_contri = contrib.max().item()
        # j_id = max_idx % 22
        # t_id = max_idx // 22
        # print(t_id,j_id, max_contri / p_attn.shape[-1])

        # for tag_id in range(176):
        # print(p_attn[0][1][tag_id])
        # max_idx = p_attn[tag_data][1][tag_id].argmax().item()
        # max_weight = round(p_attn[tag_data][1][tag_id].max().item(),2)
        # j_id = max_idx % 22
        # t_id = max_idx // 22
        # result_ele = (t_id, j_id,max_weight)
        # result_list.append(result_ele)
        # if len(result_list) == 22:
        #     print(time)
        #     print(result_list)
        #     time += 1
        #     result_list = []

        # add spatial_mask ---> only get information from different time_step
        return torch.matmul(p_attn, value), p_attn

    def forward(self, x):
        "Implements Figure 2"
        nbatches = x.size(0) # [batch, t, dim]
        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.query_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
        key = self.key_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)
        value = self.value_map(x).view(nbatches, -1, self.h_num, self.h_dim).transpose(1, 2)

        #print("query.shape:",query.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, self.use_graph,self.domain) #[batch, h_num, T, h_dim ]

            # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h_dim * self.h_num)#[batch, T, h_dim * h_num ]

        #print("att_ft.shape:", x.shape)
        #x = self.att_ft_dropout(x)
        x = self.ft_map(x)


        return x






class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, input_size, output_size, dp_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = self.w_1(x)
        return x
        #return self.dropout_2(F.relu(self.w_2(self.dropout(F.relu(self.w_1(x))))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    add relu
    """
    def __init__(self, input_size, output_size, dp_rate):
        super(SublayerConnection, self).__init__()
        #self.norm = LayerNorm(output_size)
        self.norm = LayerNorm(output_size)

        self.dropout = nn.Dropout(dp_rate)
        self.relu = nn.ReLU(inplace=True)

        if (input_size == output_size):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Linear(input_size, output_size),
                #LayerNorm(output_size),
            )


    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #act
        res = self.residual(x)
        #x = self.norm(x)
        x = sublayer(x)
        x = self.norm(x )
        x = self.relu(x) #add activation
        #x = x + res
        #x = self.norm(x)
        x = self.dropout(x)  # dropount

        #x += res


        return  x


# d_model_list = [64,128,256]
# dm_pe = {}
# for d_model in d_model_list:
#     dm_pe[d_model] = PositionalEncoding(d_model, dp_rate = 0.2, time_len = 8)

class TranformerLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, input_size, output_size, h_num, h_dim, dp_rate, time_len, use_graph,domain):
        #input_size : the dim of input
        #output_size: the dim of output
        #dropout_rate
        #use_graph: wether add graph constrain to att

        super(TranformerLayer, self).__init__()

        self.pe = PositionalEncoding(input_size, dp_rate, time_len, domain)

        self.attn = MultiHeadedAttention(h_num, h_dim, input_size, output_size, dp_rate, use_graph, domain) #do att on input dim
        self.domain = domain


        #self.feed_forward = PositionwiseFeedForward(input_size, output_size, dp_rate)# the two fc layer to output dim

        self.sublayer = nn.ModuleList([SublayerConnection(input_size, output_size, dp_rate)]) #for spatiall
                                        #SublayerConnection(input_size, output_size,dp_rate)]) #for ff
        #self.add_pe = PositionalEncoding(d_model, dp_rate=0.2, time_len=8)

        # model_list = [self.ad_pos_embed, self.self_attn,self.feed_forward, self.sublayer]
        # for model in model_list:
        #     for p in model.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform(p)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        #x = dm_pe[self.d_model](x)

        x = self.pe(x)

        x = self.sublayer[0](x, self.attn)
        return x


class Trans_Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, input_size, h_num, h_dim, output_size, dp_rate, time_len, use_graph, use_pe,domain):
        #N: the number of layers

        # input_size : the dim of input
        # output_size: the dim of output

        # h: head num
        # dropout_rate
        super(Trans_Encoder, self).__init__()

        layer = TranformerLayer(input_size, output_size, h_num, h_dim,  dp_rate, time_len, use_graph,domain)
        self.layers = nn.ModuleList([layer])
        #self.norm = LayerNorm(output_size) #use for output

        model_list = [ self.layers]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return x

