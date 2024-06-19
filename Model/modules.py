# The linear composition and isomprhismm network
# Pretrained Spectral GCN Encoder
# Classifier head
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Dropout1d


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
