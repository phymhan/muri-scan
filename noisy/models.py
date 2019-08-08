import torch
import torch.nn as nn


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


class BaseModel(nn.Module):
    """
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, num_classes=2, use_gru=False, feature_dim=31, embedding_dim=128,
                 gru_hidden_dim=32, gru_out_dim=8, dropout=0.2):
        """
        Args:
            params: dict with model parameters
        """
        super(BaseModel, self).__init__()

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
        pred = self.cls(out)
        return pred
