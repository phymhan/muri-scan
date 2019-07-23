"""Defines the neural network, loss function and metrics"""

from .layers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class Net(nn.Module):
    """
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, params):
        """
        Args:
            params: dict with model parameters
        """
        super(Net, self).__init__()

        # RNN (GRU) for global features
        if params.use_global == 0:
            self.use_global = False
            self.glb_out_dim = 0
        else:
            self.use_global = True
            self.glb_feat_dim = params.global_feature_dim
            self.glb_emb_dim = param.global_embedding_dim
            self.glb_hidden_dim = params.global_hidden_dim
            self.glb_out_dim = params.global_out_dim
            self.glb_input_map = nn.Sequential(
                nn.Linear(params.global_feature_dim, params.global_embedding_dim),
                nn.ReLU(),
                LayerNorm(params.global_embedding_dim),
                nn.Dropout(params.dropout),
            )
            self.glb_gru = nn.GRU(params.global_embedding_dim, params.global_hidden_dim)
            self.glb_gru2out = nn.Linear(params.global_hidden_dim, params.global_out_dim)

        if params.concat_global == 0:
            self.concat_global = False
            self.node_feat_dim = params.node_feature_dim
        else:
            self.concat_global = True
            self.node_feat_dim = params.node_feature_dim + params.global_feature_dim

        self.input_map = nn.Sequential(
            nn.Linear(self.node_feat_dim, params.node_embedding_dim),
            nn.ReLU(),
            LayerNorm(params.node_embedding_dim),
            nn.Dropout(params.dropout),
        )
        self.s_att = ST_ATT_Layer(input_size=params.node_embedding_dim, output_size=params.node_embedding_dim, att_heads=params.heads, latent_dim=params.latent_dim, 
                                  dp_rate=params.dropout, time_len=params.time_len, joint_num=params.joint_num, domain="spatial")
        self.t_att = ST_ATT_Layer(input_size=params.node_embedding_dim, output_size=params.node_embedding_dim, att_heads=params.heads, latent_dim=params.latent_dim, 
                                  dp_rate=params.dropout, time_len=params.time_len, joint_num=params.joint_num, domain="temporal")
        self.cls = nn.Linear(params.node_embedding_dim + self.glb_out_dim, params.num_classes)


    def glb_init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.glb_hidden_dim))
        return h.cuda(non_blocking=True)


    def forward(self, x, x_glb=None):
        # x.shape -> (batch_size, time_len, joint_num, node_feature_dim)
        # x_glb.shape -> (batch_size, time_len, global_feature_dim)
        batch_size = x.shape[0]
        time_len = x.shape[1]
        joint_num = x.shape[2]

        if self.concat_global:
            x_glb_ = x_glb.view(batch_size, time_len, 1, -1).expand(batch_size, time_len, joint_num, -1)
            x = torch.cat((x, x_glb_), 3)
        x = x.reshape(-1, time_len * joint_num, self.node_feat_dim)
        x = self.input_map(x)
        x = self.s_att(x)
        x = self.t_att(x)
        x = x.sum(1) / x.shape[1]

        x_glb = self.glb_input_map(x_glb)   # batch_size x time_len x global_embedding_dim
        if self.use_global:
            hidden = self.glb_init_hidden(batch_size)
            for i in range(time_len):
                emb = x_glb[:, i, :]                     # batch_size x global_embedding_dim
                emb = emb.view(1, -1, self.glb_emb_dim)  # 1 x batch_size x global_embedding_dim
                out, hidden = self.glb_gru(emb, hidden)
            # only use the last output
            out = self.glb_gru2out(out.view(-1, self.glb_hidden_dim))  # batch_size x global_out_dim
            x = torch.cat((x, out), 1)

        pred = self.cls(x)
        return pred


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 2 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels) / float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
