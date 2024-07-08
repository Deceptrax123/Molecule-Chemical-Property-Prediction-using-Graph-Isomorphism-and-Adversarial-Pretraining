from torch_geometric.profile import count_parameters
from Model.model_classification import MoleculePropertyClassifier
from Model.ismorphism import GCNEncoder


# Count the number of parameters
def count():
    encoder = GCNEncoder()
    model = MoleculePropertyClassifier(num_labels=1, encoder=encoder)

    params = count_parameters(model)

    return params


if __name__ == '__main__':
    param_count = count()
    print(param_count)
