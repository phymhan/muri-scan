import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import functools


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'norm':
        norm_layer = functools.partial(LayerNorm)
    elif norm_type == 'none':
        norm_layer = Identity
    else:
        raise NotImplementedError('norm layer [%s] not found' % norm_type)
    return norm_layer


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


class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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


class BaseClipModel(nn.Module):
    """
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, num_classes=2, use_gru=False, feature_dim=31, embedding_dim=128,
                 gru_hidden_dim=32, gru_out_dim=8, dropout=0.2, noisy=False):
        """
        Args:
            params: dict with model parameters
        """
        super(BaseClipModel, self).__init__()

        if not use_gru:
            self.use_gru = False
            self.input_map = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.ReLU(),
                LayerNorm(embedding_dim),
                nn.Dropout(dropout),
            )
            out_dim = embedding_dim
        else:
            self.use_gru = True
            self.feat_dim = feature_dim
            self.emb_dim = embedding_dim
            self.gru_hidden_dim = gru_hidden_dim
            self.gru_out_dim = gru_out_dim
            self.input_map = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.ReLU(),
                LayerNorm(embedding_dim),
                nn.Dropout(dropout),
            )
            self.gru = nn.GRU(embedding_dim, gru_hidden_dim)
            self.gru2out = nn.Linear(gru_hidden_dim, gru_out_dim)
            out_dim = gru_out_dim
        self.cls = nn.Linear(out_dim, num_classes)
        if noisy:
            self.transition = nn.Embedding(num_classes, num_classes)  # initialized in get_model in main
            self.noisy = True
        else:
            self.noisy = False

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.glb_hidden_dim)
        return h.cuda(non_blocking=True)

    def forward(self, x):
        # x.shape -> (batch_size, time_len, global_feature_dim)
        batch_size = x.shape[0]
        time_len = x.shape[1]

        x = self.input_map(x)   # batch_size x time_len x global_embedding_dim
        if self.use_gru == 0:
            out = torch.mean(x, 1)
        else:
            hidden = self.init_hidden(batch_size)
            for i in range(time_len):
                emb = x[:, i, :]                     # batch_size x global_embedding_dim
                emb = emb.view(1, -1, self.emb_dim)  # 1 x batch_size x global_embedding_dim
                out, hidden = self.glb_gru(emb, hidden)
            # only use the last output
            out = self.gru2out(out.view(-1, self.gru_hidden_dim))  # batch_size x global_out_dim
        pred = F.softmax(self.cls(out))
        if self.noisy:
            T = F.softmax(self.transition.weight)
            pred = torch.mm(pred, T)
        return pred


class BaseVideoModel(nn.Module):
    """
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, num_classes=2, use_gru=False, feature_dim=31, embedding_dim=128,
                 gru_hidden_dim=32, gru_out_dim=8, dropout=0.2, noisy=False):
        """
        Args:
            params: dict with model parameters
        """
        super(BaseVideoModel, self).__init__()

        if not use_gru:
            self.use_gru = False
            self.input_map = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.ReLU(),
                LayerNorm(embedding_dim),
                nn.Dropout(dropout),
            )
            out_dim = embedding_dim
        else:
            self.use_gru = True
            self.feat_dim = feature_dim
            self.emb_dim = embedding_dim
            self.gru_hidden_dim = gru_hidden_dim
            self.gru_out_dim = gru_out_dim
            self.input_map = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.ReLU(),
                LayerNorm(embedding_dim),
                nn.Dropout(dropout),
            )
            self.gru = nn.GRU(embedding_dim, gru_hidden_dim)
            self.gru2out = nn.Linear(gru_hidden_dim, gru_out_dim)
            out_dim = gru_out_dim
        self.cls = nn.Linear(out_dim, num_classes)

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.gru_hidden_dim)
        return h.cuda(non_blocking=True)

    def forward(self, x):
        # x.shape -> (batch_size, clip_num, time_len, global_feature_dim)
        batch_size = x.shape[0]
        clip_num = x.shape[1]
        time_len = x.shape[2]
        time_len *= clip_num
        x = self.input_map(x)
        x = x.view(batch_size, time_len, -1)  # batch_size x time_len x global_embedding_dim)
        if not self.use_gru:
            out = torch.mean(x, 1)
        else:
            hidden = self.init_hidden(batch_size)
            for t in range(time_len):
                emb = x[:, t, :]                     # batch_size x global_embedding_dim
                emb = emb.view(1, -1, self.emb_dim)  # 1 x batch_size x global_embedding_dim
                out, hidden = self.gru(emb, hidden)
            # only use the last output
            out = self.gru2out(out.view(-1, self.gru_hidden_dim))  # batch_size x global_out_dim
        pred = self.cls(out)
        return F.softmax(pred)


class BaseVideoModelV2(nn.Module):
    """
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, num_classes=2, use_gru=False, feature_dim=31, embedding_dim=128,
                 gru_hidden_dim=32, gru_out_dim=8, dropout=0.2, noisy=False,
                 dim_input_map=[128], norm_input_map='none', dim_fc=[], norm_fc='none'):
        """
        Args:
            params: dict with model parameters
        """
        super(BaseVideoModelV2, self).__init__()
        norm_input = get_norm_layer(norm_input_map)
        norm_fc = get_norm_layer(norm_fc)
        blocks = []
        f0 = feature_dim
        for f1 in dim_input_map:
            blocks += [nn.Linear(f0, f1),
                       nn.ReLU(),
                       norm_input(f1),
                       nn.Dropout(dropout)]
            f0 = f1
        self.input_map = nn.Sequential(*blocks)
        if not use_gru:
            self.use_gru = False
            out_dim = f1
        else:
            self.use_gru = True
            self.feat_dim = feature_dim
            self.emb_dim = f1
            self.gru_hidden_dim = gru_hidden_dim
            self.gru_out_dim = gru_out_dim
            self.gru = nn.GRU(f1, gru_hidden_dim)
            self.gru2out = nn.Linear(gru_hidden_dim, gru_out_dim)
            out_dim = gru_out_dim
        blocks = []
        f0 = out_dim
        f1 = out_dim
        for f1 in dim_fc:
            blocks += [nn.Linear(f0, f1),
                       nn.ReLU(),
                       norm_fc(f1),
                       nn.Dropout(dropout)]
            f0 = f1
        if blocks:
            self.fc = nn.Sequential(*blocks)
        else:
            self.fc = None
        self.cls = nn.Linear(f1, num_classes)
        if noisy:
            self.transition = nn.Embedding(num_classes, num_classes)  # initialized in get_model in main
            self.noisy = True
        else:
            self.noisy = False

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.gru_hidden_dim)
        return h.cuda(non_blocking=True)

    def forward(self, x):
        # x.shape -> (batch_size, clip_num, time_len, global_feature_dim)
        batch_size = x.shape[0]
        clip_num = x.shape[1]
        time_len = x.shape[2]
        x = self.input_map(x)  # batch_size x clip_num x time_len x global_embedding_dim)
        if self.use_gru == 0:
            out = torch.mean(x, 2)  # batch_size x clip_num x global_embedding_dim)
        else:
            hidden = self.init_hidden(batch_size*clip_num)
            for t in range(time_len):
                emb = x[:, :, t, :]                  # batch_size x clip_num x global_embedding_dim
                emb = emb.view(1, -1, self.emb_dim)  # 1 x batch_size*clip_num x global_embedding_dim
                out, hidden = self.gru(emb, hidden)
            # only use the last output
            out = self.gru2out(out.view(-1, self.gru_hidden_dim))  # batch_size*clip_num x global_out_dim
            out = out.view(batch_size, clip_num, -1)
        # clip feat -> video feat
        out = torch.mean(out, 1)
        if self.fc:
            out = self.fc(out)
        pred = self.cls(out)
        return F.softmax(pred)


class WeaklyVideoModel(nn.Module):
    """
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, num_classes=2, use_gru=False, feature_dim=31, embedding_dim=128,
                 gru_hidden_dim=32, gru_out_dim=8, dropout=0.2, noisy=False,
                 dim_input_map=[128], norm_input_map='none', dim_fc=[], norm_fc='none'):
        """
        Args:
            params: dict with model parameters
        """
        super(WeaklyVideoModel, self).__init__()
        norm_input = get_norm_layer(norm_input_map)
        norm_fc = get_norm_layer(norm_fc)
        blocks = []
        f0 = feature_dim
        for f1 in dim_input_map:
            blocks += [nn.Linear(f0, f1),
                       nn.ReLU(),
                       norm_input(f1),
                       nn.Dropout(dropout)]
            f0 = f1
        self.input_map = nn.Sequential(*blocks)
        if not use_gru:
            self.use_gru = False
            out_dim = f1
        else:
            self.use_gru = True
            self.feat_dim = feature_dim
            self.emb_dim = f1
            self.gru_hidden_dim = gru_hidden_dim
            self.gru_out_dim = gru_out_dim
            self.gru = nn.GRU(f1, gru_hidden_dim)
            self.gru2out = nn.Linear(gru_hidden_dim, gru_out_dim)
            out_dim = gru_out_dim
        blocks = []
        f0 = out_dim
        f1 = out_dim
        for f1 in dim_fc:
            blocks += [nn.Linear(f0, f1),
                       nn.ReLU(),
                       norm_fc(f1),
                       nn.Dropout(dropout)]
            f0 = f1
        if blocks:
            self.fc = nn.Sequential(*blocks)
        else:
            self.fc = None
        self.fc1c = nn.Linear(f1, num_classes)
        self.fc1d = nn.Linear(f1, num_classes)
        if noisy:
            self.transition = nn.Embedding(num_classes, num_classes)  # initialized in get_model in main
            self.noisy = True
        else:
            self.noisy = False

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.gru_hidden_dim)
        return h.cuda(non_blocking=True)

    def init_transition(self):
        self.transition.weight.data.copy_(torch.tensor([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])*5.)

    def compute_trace(self):
        return torch.trace(self.transition.weight)

    def forward(self, x):
        # x.shape -> (batch_size, clip_num, time_len, global_feature_dim)
        batch_size = x.shape[0]
        clip_num = x.shape[1]
        time_len = x.shape[2]
        x = self.input_map(x)  # batch_size x clip_num x time_len x global_embedding_dim)
        if self.use_gru == 0:
            out = torch.mean(x, 2)  # batch_size x clip_num x global_embedding_dim)
        else:
            hidden = self.init_hidden(batch_size*clip_num)
            for t in range(time_len):
                emb = x[:, :, t, :]                  # batch_size x clip_num x global_embedding_dim
                emb = emb.view(1, -1, self.emb_dim)  # 1 x batch_size*clip_num x global_embedding_dim
                out, hidden = self.gru(emb, hidden)
            # only use the last output
            out = self.gru2out(out.view(-1, self.gru_hidden_dim))  # batch_size*clip_num x global_out_dim
            out = out.view(batch_size, clip_num, -1)
        if self.fc:
            out = self.fc(out)
        # weakly
        sigma_c = F.softmax(self.fc1c(out), 2)  # batch_size x clip_num x num_classes
        if self.noisy:
            # pdb.set_trace()
            T = F.softmax(self.transition.weight)
            sigma_c = torch.mm(sigma_c.view(batch_size*clip_num, -1), T).view(batch_size, clip_num, -1)
        sigma_d = F.softmax(self.fc1d(out), 1)  # batch_size x clip_num x num_classes
        x = sigma_c * sigma_d
        pred = torch.sum(x, dim=1)
        return pred


class WeaklyVideoModelV2(nn.Module):
    """
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, num_classes=2, use_gru=False, feature_dim=31, embedding_dim=128,
                 gru_hidden_dim=32, gru_out_dim=8, dropout=0.2, noisy=False,
                 dim_input_map=[128], norm_input_map='none', dim_fc=[], norm_fc='none', video_pooling='avg', dim_fc_noisy=[]):
        """
        Args:
            params: dict with model parameters
        """
        super(WeaklyVideoModelV2, self).__init__()
        self.video_pooling = video_pooling
        self.num_classes = num_classes
        norm_input = get_norm_layer(norm_input_map)
        norm_fc = get_norm_layer(norm_fc)
        blocks = []
        f0 = feature_dim
        for f1 in dim_input_map:
            blocks += [nn.Linear(f0, f1),
                       nn.ReLU(),
                       norm_input(f1),
                       nn.Dropout(dropout)]
            f0 = f1
        self.input_map = nn.Sequential(*blocks)
        if not use_gru:
            self.use_gru = False
            out_dim = f1
        else:
            self.use_gru = True
            self.feat_dim = feature_dim
            self.emb_dim = f1
            self.gru_hidden_dim = gru_hidden_dim
            self.gru_out_dim = gru_out_dim
            self.gru = nn.GRU(f1, gru_hidden_dim)
            self.gru2out = nn.Linear(gru_hidden_dim, gru_out_dim)
            out_dim = gru_out_dim
        blocks = []
        f0 = out_dim
        f1 = out_dim
        for f1 in dim_fc[:-1]:
            blocks += [nn.Linear(f0, f1),
                       nn.ReLU(),
                       norm_fc(f1),
                       nn.Dropout(dropout)]
            f0 = f1
        f1 = dim_fc[-1]
        blocks += [nn.Linear(f0, f1)]
        if blocks:
            self.fc = nn.Sequential(*blocks)
        else:
            self.fc = None
        self.fc1c = nn.Linear(f1, num_classes)
        self.fc1d = nn.Linear(f1, num_classes)
        if noisy:
            if not dim_fc_noisy:
                dim_fc_noisy = [num_classes ** 2]
            blocks = []
            f0 = out_dim
            f1 = out_dim
            for f1 in dim_fc_noisy[:-1]:
                blocks += [nn.Linear(f0, f1),
                           nn.ReLU(),
                           norm_fc(f1),
                           nn.Dropout(dropout)]
                f0 = f1
            f1 = dim_fc_noisy[-1]
            blocks += [nn.Linear(f0, f1)]
            self.transition = nn.Sequential(*blocks)
            self.noisy = True
        else:
            self.noisy = False

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.gru_hidden_dim)
        return h.cuda(non_blocking=True)

    def init_transition(self):
        self.transition[-1].weight.data.zero_()
        self.transition[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0.5, 0.5, 0], dtype=torch.float)*5.)

    def compute_trace(self):
        return torch.mean(torch.sum(torch.diagonal(self.T, dim1=1, dim2=2), 1))

    def forward(self, x):
        # x.shape -> (batch_size, clip_num, time_len, global_feature_dim)
        batch_size = x.shape[0]
        clip_num = x.shape[1]
        time_len = x.shape[2]
        x = self.input_map(x)  # batch_size x clip_num x time_len x global_embedding_dim)
        if self.use_gru == 0:
            out = torch.mean(x, 2)  # batch_size x clip_num x global_embedding_dim)
        else:
            hidden = self.init_hidden(batch_size*clip_num)
            for t in range(time_len):
                emb = x[:, :, t, :]                  # batch_size x clip_num x global_embedding_dim
                emb = emb.view(1, -1, self.emb_dim)  # 1 x batch_size*clip_num x global_embedding_dim
                out, hidden = self.gru(emb, hidden)
            # only use the last output
            out = self.gru2out(out.view(-1, self.gru_hidden_dim))  # batch_size*clip_num x global_out_dim
            out = out.view(batch_size, clip_num, -1)

        # clip feat -> video feat
        if self.video_pooling == 'avg':
            vid_feat = torch.mean(out, 1)
        elif self.video_pooling == 'max':
            vid_feat = torch.max(out, 1)
        else:
            raise NotImplementedError

        if self.fc:
            out = self.fc(out)
        # weakly
        sigma_c = F.softmax(self.fc1c(out), 2)  # batch_size x clip_num x num_classes
        if self.noisy:
            T = self.transition(vid_feat)  # batch_size x num_classes*num_classes
            T = F.softmax(T.view(-1, self.num_classes, self.num_classes), 2)
            sigma_c = torch.matmul(sigma_c.view(batch_size, clip_num, 1, self.num_classes),
                                   T.view(batch_size, 1, self.num_classes, self.num_classes))\
                .view(batch_size, clip_num, self.num_classes)
            self.T = T
        sigma_d = F.softmax(self.fc1d(out), 1)  # batch_size x clip_num x num_classes
        x = sigma_c * sigma_d
        pred = torch.sum(x, dim=1)
        return pred
