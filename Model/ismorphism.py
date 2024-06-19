# The linear composition and isomprhismm network
# Pretrained Spectral GCN Encoder
# Classifier head
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Dropout1d
from torch_geometric.nn import GINConv, ChebConv
import torch.nn.functional as F


class GCNEncoder(Module):
    def __init__(self):
        super(GCNEncoder, self).__init__()


class GraphIsomorphismNetwork(Module):
    def __init__(self):
        super(GraphIsomorphismNetwork, self).__init__()
        self.gin1 = GINConv(nn=PhiLinearMapping(86, 128))
        self.gin2 = GINConv(nn=PhiLinearMapping(128, 256))
        self.gin3 = GINConv(nn=PhiLinearMapping(256, 512))

    def forward(self, v, edges):
        x1 = F.relu(self.gin1(v, edge_index=edges))
        x2 = F.relu(self.gin2(x1, edge_index=edges))
        x3 = F.relu(self.gin3(x2, edge_index=edges))

        return x1, x2, x3


class PhiLinearMapping(Module):
    def __init__(self, in_channels, out_channels):
        super(PhiLinearMapping, self).__init__()
        self.linear = Linear(in_features=in_channels,
                             out_features=out_channels)
        self.relu = ReLU()
        self.bn = BatchNorm1d(num_features=out_channels)
        self.dropout = Dropout1d(p=0.3)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
