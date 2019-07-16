import torchvision
import torch.nn as nn
import torch
from .gcn_layer import GraphConvolution
import numpy as np




class Partition_layer(nn.Module):
    def __init__(self, input_size, output_size, use_residual, use_bn, use_edge_weight,dp_rate):
        super(Partition_layer, self).__init__()

        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_edge_weight= use_edge_weight

        #residual
        self.res_bn = None
        if self.use_residual:
            if (input_size == output_size):
                self.residual = lambda x: x

            else:
                self.residual = nn.Sequential(
                    nn.Linear(input_size, output_size),
                )
                self.initialize_weights(self.residual)

                if self.use_bn:
                    self.res_bn = nn.BatchNorm1d(output_size)
                    self.initialize_weights(self.res_bn)

        #bn

        if self.use_bn:
            self.s_bn = nn.BatchNorm1d(output_size)
            self.initialize_weights(self.s_bn)

            self.t_bn = nn.BatchNorm1d(output_size)
            self.initialize_weights(self.t_bn)


        self.joint_num = 22

        #edge_weight
        if self.use_edge_weight :
            #self.edg_weight = nn.parameter.Parameter(torch.ones(self.joint_num, self.joint_num * 3)) # [t-1, t, t + 1]
            self.edg_weight = nn.parameter.Parameter(torch.ones(self.joint_num, self.joint_num))
            self.softmax = nn.Softmax(dim=2)

        #layer
            #spatial
        self.s_layer = GraphConvolution(input_size, output_size)
        self.I_s_layer = GraphConvolution(input_size, output_size)
            #temporal
        self.prev_layer = GraphConvolution(output_size, output_size)
        self.next_layer = GraphConvolution(output_size, output_size)
        self.I_t_layer = GraphConvolution(output_size, output_size,)

        self.dropout = nn.Dropout(dp_rate)

        self.s_relu = nn.ReLU()
        self.t_relu = nn.ReLU()



    def forward(self, x, adj):
        #x [batch, node_num, ft]

        #calculate_residual
        res = 0
        if self.use_residual:
            res = self.residual(x)
            if self.res_bn:
                res = self.apply_bn(res,self.res_bn)


        #temporal_adj
        prev_adj = adj[1]
        next_adj = adj[2]
        I_t = adj[3]

        #spatial adj
        if self.use_edge_weight:
            adj = self.use_time_invariant_weight(adj)
        s_adj = adj[0]
        I_s = adj[1]


        #spatial_forward
        #print(x.shape)
        s_ft = self.s_layer(x, s_adj)
        I_s_ft = self.I_s_layer(x, I_s)
        x = s_ft + I_s_ft
        #print(x.shape)
        if self.use_bn:
            x = self.apply_bn(x,self.s_bn)
        #print(x.shape)
        x = self.s_relu(x)
        #print(x.shape)


        #temporal_ft
        I_t_ft = self.I_t_layer(x,I_t)
        prev_ft = self.prev_layer(x, prev_adj)
        next_ft = self.next_layer(x, next_adj)
        x = I_t_ft + prev_ft + next_ft
        if self.use_bn:
            x = self.apply_bn(x, self.t_bn)
        x = self.dropout(x)

        return self.t_relu(x  + res)


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




    def apply_bn(self, feature, bn_layer):

        # print("before reshape")#[batch,video_length,ft_dim]
        # print(feature[0, 0, 0:3])
        # print(feature[1,154,0:3])

        #
        feature = feature.permute(0, 2, 1)#[batch,ft_dim,video_length]

        feature = bn_layer(feature)

        feature = feature.permute(0, 2, 1)#batch, video_length, ft_Dim

        return feature




    def use_time_invariant_weight(self, adj):
        #adj [batch_size, T * joint_num, T * joint_num ]
        A = adj[0]
        I = adj[3]


        combine_adj = A + I

        node_num = combine_adj.shape[1]
        T = int(node_num / self.joint_num)
        new_adj = torch.zeros(combine_adj.shape).cuda()
        for t_ele in  range(T):

            #(0,0)  --> (joint_num, joint_num) -- > (joint_num * 2, joint_num * 2) --> ...... -->(joint_num * （t - 1）, joint_num * （t - 1）)
            row_begin = t_ele * self.joint_num
            row_num = self.joint_num

            column_begin = row_begin
            column_num = row_num


            temp_adj = combine_adj[:, row_begin : row_begin + row_num, column_begin : column_begin + column_num]

            new_adj[:, row_begin : row_begin + row_num, column_begin : column_begin + column_num] = temp_adj * self.edg_weight


        norm_mask = (1 - combine_adj) * (-9e15)
        new_adj = new_adj + norm_mask
        new_adj = self.softmax(new_adj)

        #print(new_adj[0, 0:self.joint_num * 2, 0:self.joint_num * 2])

        new_A = A * new_adj
        new_I = I * new_adj

        return [new_A, new_I]

#





class GCN_branch(nn.Module):
    def __init__(self, input_size, output_size,dp_rate):
        super(GCN_branch, self).__init__()

        #self.lp_1 = LP_GCN(512,2048)
        #self.lp_2 = LP_GCN(2048,12)

        #self.gcn_network = GCN_RES(input_size, 64)
        #print(output_size)
        self.use_bn = True
        self.use_edge_weight = True
        self.gcn_network = nn.ModuleList([

            # Partition_layer(input_size, 64, use_residual = False, use_edge_weight = self.use_edge_weight, use_bn = self.use_bn),
            # Partition_layer(64, 64, use_residual = True, use_edge_weight = self.use_edge_weight,  use_bn = self.use_bn),
            # Partition_layer(64, 64, use_residual = True, use_edge_weight = self.use_edge_weight,  use_bn = self.use_bn),
            # Partition_layer(64, 64, use_residual = True, use_edge_weight = self.use_edge_weight,  use_bn = self.use_bn),
            # Partition_layer(64, 128, use_residual=True, use_edge_weight=self.use_edge_weight, use_bn=self.use_bn),
            #
            # Partition_layer(128, 128, use_residual = True, use_edge_weight = self.use_edge_weight,  use_bn = self.use_bn),
            # Partition_layer(128, 128, use_residual = True, use_edge_weight = self.use_edge_weight,  use_bn = self.use_bn),
            # Partition_layer(128, 256, use_residual = True, use_edge_weight=self.use_edge_weight, use_bn=self.use_bn),
            #
            # Partition_layer(256, 256, use_residual=True, use_edge_weight=self.use_edge_weight, use_bn=self.use_bn),
            # Partition_layer(256, 256, use_residual = True, use_edge_weight = self.use_edge_weight,  use_bn = self.use_bn),

            # 3 --> 64
            Partition_layer(input_size, 128, use_residual=False, use_edge_weight=self.use_edge_weight,
                            use_bn=self.use_bn, dp_rate = dp_rate),
            # 64 --> 128
            Partition_layer(128, 128, use_residual=True, use_edge_weight=self.use_edge_weight, use_bn=self.use_bn, dp_rate = dp_rate),
            # 128 --> 256
            Partition_layer(128, 128, use_residual=True, use_edge_weight=self.use_edge_weight, use_bn=self.use_bn, dp_rate = dp_rate),



            ]
        )
        print("\n**************************************************")
        print("use_bn:", self.use_bn, "use_edge_weight:", self.use_edge_weight, "layer_num:",len(self.gcn_network))
        print("**************************************************")



    def forward(self, cat_feature, adj):

        for layer_ele in  self.gcn_network:

            cat_feature = layer_ele(cat_feature,adj)
            #print("layer output")
            #print(cat_feature[0][0][0:3])
            #print(cat_feature[1][154][0:3])

        #print(cat_feature[0][0][:3])
        #print(cat_feature[0][-22][:3])



        cat_feature = cat_feature.sum(1) / cat_feature.shape[1]




        return cat_feature



class Invweight_4_part(nn.Module):
    #send the locations sequence of each joint to FC first to encode motion information first
    #use gcn to fuse motion information and make prediction

    def __init__(self, num_classes,dp_rate):
        super(Invweight_4_part, self).__init__()
        self.feature_size = 3
        self.joints_num = 22
        self.T = 8

        #self.input_bn = nn.BatchNorm1d(self.feature_size)
        #self.initialize_weights(self.input_bn)

        self.gcn_ft_dim = 128


        self.gcn_encoder = GCN_branch(self.feature_size, self.gcn_ft_dim,dp_rate )

        self.cls = nn.Linear(self.gcn_ft_dim, num_classes)
        self.initialize_weights(self.cls)


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



    def forward(self, x, adj):

        # x [batch_size, seq,  22, 3]
        seq_num = x.shape[1]
        #batch_size = x.shape[0]

        #normalize input
        #print("x.shape:",x.shape)
        #x = x.view(batch_size, 3, seq_num * self.joints_num)
        #x = self.bn_1(x)
        #x = x.view(self.joints_num, seq_num, batch_size, 3)
        #x = x.view(batch_size, seq_num, self.joints_num, 3)

        for t in range(seq_num):
            x_t = x[:, t, :, :] #[batch,22,3]
            if t == 0:
                gcn_input = x_t
            else:
                gcn_input = torch.cat((gcn_input,x_t), 1)

        # gcn_input = gcn_input.permute(0, 2, 1)  # [batch, 3, 22*8]
        # gcn_input = self.input_bn(gcn_input)
        # gcn_input = gcn_input.permute(0, 2, 1)


        x = self.gcn_encoder(gcn_input, adj)
        #x = joint_motion.sum(1)
        #x = joint_motion.sum(1)
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




