from torch.nn import Module, Linear, ReLU, BatchNorm1d, Dropout1d
from torch_geometric.nn import global_mean_pool
from Model.ismorphism import GraphIsomorphismNetwork
import torch.nn.functional as F


class MoleculePropertyClassifier(Module):
    def __init__(self, num_labels, encoder):
        super(MoleculePropertyClassifier, self).__init__()

        self.nn = GraphIsomorphismNetwork(encoder=encoder)
        self.linear1 = Linear(in_features=512, out_features=256)
        self.classifier = Linear(in_features=256, out_features=num_labels)

        self.relu = ReLU()
        self.bn = BatchNorm1d(num_features=256)

    def forward(self, v, edges, batch):
        v = self.nn.forward(v, edges)
        v = global_mean_pool(v, batch=batch)  # Graph View

        v = self.linear1(v)
        v = self.bn(v)
        v = self.relu(v)

        v = self.classifier(v)

        return v, F.sigmoid(v)
