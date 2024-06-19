from torch.nn import Module
from torch_geometric.nn import global_mean_pool
from Model.ismorphism import GraphIsomorphismNetwork
import torch.nn.functional as F


class MoleculePropertyClassifier(Module):
    def __init__(self):
        super(MoleculePropertyClassifier, self).__init__()

        self.nn = GraphIsomorphismNetwork()

    def forward(self, v, edges, batch):
        v = self.nn(v, edge_index=edges)
        v = global_mean_pool(v, batch=batch)

        return v, F.sigmoid(v)
