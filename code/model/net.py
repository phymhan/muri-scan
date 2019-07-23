"""Defines the neural network, loss function and metrics"""

from .layers import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.input_map = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(params.dropout),
        )
        self.s_att = ST_ATT_Layer(input_size=128, output_size=128, att_heads=params.heads, latent_dim=params.latent_dim, 
                                    dp_rate=params.dropout, time_len=params.time_len, joint_num=params.joint_num, domain="spatial"
                                )
        self.t_att = ST_ATT_Layer(input_size=128, output_size=128, att_heads=params.heads, latent_dim=params.latent_dim, 
                                    dp_rate=params.dropout, time_len=params.time_len, joint_num=params.joint_num, domain="temporal"
                                )
        self.cls = nn.Linear(128, params.num_classes)


    def forward(self, x):
        # x.shape -> (batch_size, time_len, joint_num, 3)
        time_len = x.shape[1]
        joint_num = x.shape[2]
        x = x.reshape(-1, time_len * joint_num, 3)

        x = self.input_map(x)
        x = self.s_att(x)
        x = self.t_att(x)
        x = x.sum(1) / x.shape[1]
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
